# Copyright 2025 TeleAI-infra Team and HuggingFace Inc. All rights reserved.

import os
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from megatron.core.transformer.attention import (
    CrossAttention,
    CrossAttentionSubmodules,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from teletron.models.dit.dit_attention import (
    JointSelfAttentionSubmodules,
    JointHunyuanAttention,
    HunyuanSingleAttention,
)

from teletron.models.dit.dit_fusedlayers import (
    FusedAdaLayerNormZero,
    FusedAdaLayerNormZeroSingle,
    Get_RMSNorm
)
RMSNorm = Get_RMSNorm()



from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)





class AdaLNContinuous(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        conditioning_embedding_dim: int,
        modulation_bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(conditioning_embedding_dim, config.hidden_size * 2, bias=modulation_bias)
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6, bias=modulation_bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(config.hidden_size, eps=1e-6)
        else:
            raise ValueError("Unknown normalization type {}".format(norm_type))

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.adaLN_modulation(conditioning_embedding)
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class HunyuanDiTLayer(TransformerLayer):
    """A double transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    HunyuanDiTLayer layer implementation from [https://arxiv.org/pdf/2403.03206].
    """
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        fused_kernels: bool = False,
        context_pre_only: bool = False,
    ):
        hidden_size = config.hidden_size
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)


        self.norm1 = FusedAdaLayerNormZero(hidden_size, norm_type="layer_norm", fused_kernels=fused_kernels, config=config)
        self.norm1_context= FusedAdaLayerNormZero(hidden_size, norm_type="layer_norm", fused_kernels=fused_kernels, config=config)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        cp_override_config = copy.deepcopy(config)
        cp_override_config.context_parallel_size = 1
        cp_override_config.tp_comm_overlap = False

        self.ff_context=build_module(
            submodules.mlp,
            config=cp_override_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin:  Optional[torch.Tensor] = None,
    ):
        # 1. Input normalization
        temb = temb.contiguous()
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        
        # 2. Joint attention
        attn_output, context_attn_output = self.self_attention(
            # hidden_states=norm_hidden_states,
            # additional_hidden_states=norm_encoder_hidden_states,
            # attention_mask=attention_mask,
            # rotary_pos_emb=freqs_cis,
            norm_hidden_states, # [2,9604, 3072]
            attention_mask=attention_mask,  
            key_value_states=None,
            additional_hidden_states=norm_encoder_hidden_states,    # [2, 226, 3072]
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        # #sbd -> bsd
        # attn_output = attn_output.transpose(0, 1)
        # context_attn_output = context_attn_output.transpose(0, 1)

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        norm_hidden_states = norm_hidden_states.transpose(0, 1)
        norm_encoder_hidden_states = norm_encoder_hidden_states.transpose(0, 1)

        # 4. Feed-forward
        ff_output = self.mlp(norm_hidden_states)

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if len(ff_output)==2: 
            ff=(ff_output[0]+ff_output[1]).transpose(0, 1)
        if len(context_ff_output)==2: 
            context_ff=(context_ff_output[0]+context_ff_output[1]).transpose(0, 1)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff
        )

        return hidden_states, encoder_hidden_states
    

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if self.config.num_moe_experts is not None:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


class HunyuanSingleDiTLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        mlp_ratio: int = 4,
        fused_kernels: bool = False,
        n_adaln_chunks: int = 3,
        modulation_bias: bool = True,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number)
        hidden_size = config.hidden_size

        self.norm = FusedAdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm", fused_kernels=fused_kernels, config=config)

        self.mlp_hidden_dim=hidden_size*mlp_ratio
        # self.proj_mlp=nn.Linear(hidden_size, self.mlp_hidden_dim)
        self.proj_mlp=TEColumnParallelLinear(
                hidden_size,
                self.mlp_hidden_dim,
                config=config,
                gather_output=False,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                is_expert=False)
        
        self.act_mlp=nn.GELU(approximate="tanh")
        # self.proj_out =nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        # self.proj_out=TEColumnParallelLinear(
        #         hidden_size + self.mlp_hidden_dim,
        #         hidden_size,
        #         config=config,
        #         gather_output=False,
        #         init_method=config.init_method,
        #         bias=True,
        #         skip_bias_add=False,
        #         is_expert=False)
        
        self.proj_out = TERowParallelLinear(
                hidden_size + self.mlp_hidden_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='proj',
        )
    
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = temb.contiguous()
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        x = self.proj_mlp(norm_hidden_states)
        # print(x[0])
        # mlp_hidden_states = self.act_mlp(x)
        mlp_hidden_states = self.act_mlp(x[0])
        # mlp_hidden_states = gather_from_tensor_model_parallel_region(mlp_hidden_states)

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.self_attention(
            hidden_states=norm_hidden_states,
            additional_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )

        #sbd -> bsd
        # attn_output = attn_output.transpose(0, 1)
        # context_attn_output = context_attn_output.transpose(0, 1)
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        # attn_output = gather_from_tensor_model_parallel_region(attn_output)
        # 3. Modulation and residual connection
        
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        x, bias = self.proj_out(hidden_states)
        # print(f"x:{x[0].shape}")
        # x = gather_from_tensor_model_parallel_region(x[0])

        hidden_states = gate.unsqueeze(1) * x
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states

    
    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer block.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
                Defaults to an empty string.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (dict, optional): Additional metadata for sharding.
                Can specify if layers are non-homogeneous. Defaults to None.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the model.
        """
        assert not sharded_offsets, "Unexpected sharded offsets"
        non_homogeneous_layers = metadata is not None and metadata.get(
            'non_homogeneous_layers', False
        )
        if self.config.num_moe_experts is not None:
            non_homogeneous_layers = True

        sharded_state_dict = {}

        layer_prefix = f'{prefix}layers.'
        num_layers = self.config.num_layers
        for layer in self.layers:
            offset = TransformerLayer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = f'{layer_prefix}{global_layer_offset - offset}.'  # module list index in TransformerBlock # pylint: disable=line-too-long
            if non_homogeneous_layers:
                sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
                sharded_pp_offset = []
            else:
                sharded_prefix = layer_prefix
                sharded_pp_offset = [
                    (0, global_layer_offset, num_layers)
                ]  # PP sharding offset for ShardedTensors
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module, f'{prefix}{name}.', sharded_offsets, metadata
                    )
                )

        return sharded_state_dict


def get_hunyuan_double_transformer_engine_spec() -> ModuleSpec:
    return ModuleSpec(
        module=HunyuanDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=JointHunyuanAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=JointSelfAttentionSubmodules(
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    added_q_layernorm=RMSNorm,
                    added_k_layernorm=RMSNorm,
                    linear_qkv=TEColumnParallelLinear,
                    added_linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    #dropout？dropout=0
                    linear_fc2=TERowParallelLinear,
                ),
            ),
        ),
    )

def get_hunyuan_single_transformer_engine_spec() -> ModuleSpec:
   return ModuleSpec(
        module=HunyuanSingleDiTLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=HunyuanSingleAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    q_layernorm=RMSNorm,
                    k_layernorm=RMSNorm,
                    linear_proj=IdentityOp,
                ),
            ),
        ),
    )