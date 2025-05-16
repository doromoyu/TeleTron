# Copyright 2025 TeleAI-infra Team and HuggingFace Inc. All rights reserved.

import os
from typing import Callable,Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core import mpu, tensor_parallel
from teletron.core.context_parallel.pad import pad_for_context_parallel, remove_pad_for_context_parallel, set_origin_length, set_target_length
from torch import nn

from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from teletron.models.dit.dit_layerspec import (
    AdaLNContinuous,
    HunyuanSingleDiTLayer,
    HunyuanDiTLayer,
    get_hunyuan_double_transformer_engine_spec,
    get_hunyuan_single_transformer_engine_spec,
)

from teletron.models.hunyuanvideo.layers import HunyuanVideoPatchEmbed, HunyuanVideoTokenRefiner, CombinedTimestepGuidanceTextProjEmbeddings,HunyuanVideoRotaryPosEmbed
from teletron.core.tensor_parallel.mappings import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class HunyuanParams:
    hidden_size: int = 3072
    num_attention_heads: int = 24
    activation_func: Callable = F.gelu
    add_qkv_bias: bool = True
    in_channels: int = 33
    out_channels: int = 16
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 3
    num_single_layers: int = 6
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 2
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    guidance_embeds: bool = True
    text_embed_dim: int = 4096
    pooled_projection_dim: int = 768
    rope_theta: float = 256.0
    rope_axes_dim: Tuple[int] = (16, 56, 56)

class HunyuanVideoTransformer3DModel(VisionModule):
    def __init__(self, hunyuan_config: HunyuanParams, config: TransformerConfig):
        self.out_channels = hunyuan_config.out_channels
        self.in_channels = hunyuan_config.in_channels
        self.num_attention_heads = hunyuan_config.num_attention_heads
        self.attention_head_dim = hunyuan_config.attention_head_dim
        self.num_layers = hunyuan_config.num_layers
        self.num_single_layers = hunyuan_config.num_single_layers
        self.fused_kernels = False
        if os.environ.get("FUSED_KERNELS"):
            fused_kernels = bool(int(os.environ.get("FUSED_KERNELS")))
            self.fused_kernels = fused_kernels

        if os.environ.get("NUM_LAYERS"):
            try:
                num_layers = int(os.environ.get("NUM_LAYERS"))
                assert isinstance(num_layers, int)
                self.num_layers = num_layers
            except ValueError:
                raise ValueError(f"Invalid integer value for NUM_LAYERS: {os.environ.get('NUM_LAYERS')}")

        if os.environ.get("NUM_SINGLE_LAYERS"):
            try:
                num_single_layers = int(os.environ.get("NUM_SINGLE_LAYERS"))
                assert isinstance(num_single_layers, int)
                self.num_single_layers = num_single_layers
            except ValueError:
                raise ValueError(f"Invalid integer value for NUM_SINGLE_LAYERS: {os.environ.get('NUM_SINGLE_LAYERS')}")
        self.num_refiner_layers = hunyuan_config.num_refiner_layers
        self.mlp_ratio = hunyuan_config.mlp_ratio
        self.patch_size = hunyuan_config.patch_size
        self.patch_size_t = hunyuan_config.patch_size_t
        self.qk_norm = hunyuan_config.qk_norm
        self.text_embed_dim = hunyuan_config.text_embed_dim
        self.pooled_projection_dim = hunyuan_config.pooled_projection_dim
        self.rope_theta = hunyuan_config.rope_theta
        self.rope_axes_dim = hunyuan_config.rope_axes_dim
        self.guidance_embed = hunyuan_config.guidance_embeds

        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        config.hidden_size =self.hidden_size
        config.num_attention_heads=self.num_attention_heads

        config.num_query_groups = config.num_attention_heads
        config.use_cpu_initialization = True
        config.activation_func =hunyuan_config.activation_func
        config.hidden_dropout=0
        config.attention_dropout=0
        config.layernorm_epsilon=1e-6
        config.add_qkv_bias=hunyuan_config.add_qkv_bias
        config.rotary_interleaved=True
        config.attention_dropout = config.attention_dropout[0] if isinstance(config.attention_dropout, tuple) else config.attention_dropout
        transformer_config = config

        super().__init__(transformer_config)
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        
        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed(
            (self.patch_size_t, self.patch_size, self.patch_size), self.in_channels, self.inner_dim
        )
        self.context_embedder = HunyuanVideoTokenRefiner(
            self.text_embed_dim,
            self.num_attention_heads,
            self.attention_head_dim,
            num_layers=self.num_refiner_layers,
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            self.inner_dim, self.pooled_projection_dim
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(
            self.patch_size, self.patch_size_t, self.rope_axes_dim, self.rope_theta
        )

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanDiTLayer(
                    config=transformer_config,
                    submodules=get_hunyuan_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    fused_kernels=self.fused_kernels
                )
                for i in range(self.num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanSingleDiTLayer(
                    config=transformer_config,
                    submodules=get_hunyuan_single_transformer_engine_spec().submodules,
                    layer_number=i,
                    fused_kernels=self.fused_kernels
                )
                for i in range(self.num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLNContinuous(
            config=transformer_config, conditioning_embedding_dim=self.hidden_size
        )
        self.proj_out = nn.Linear(
            self.inner_dim, self.patch_size_t * self.patch_size * self.patch_size * self.out_channels
        )

        self.gradient_checkpointing = False
        
        print("HunyuanVideoTransformer3DModel Init Finish!")
    
    def _get_block(
            self,
            dit_type: str,
            layer_number: int
    ):
        if dit_type == "double_stream":
            return self.transformer_blocks[layer_number]
        elif dit_type == "single_stream":
            return self.single_transformer_blocks[layer_number]
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
    
    def _checkpointed_forward(
            self, 
            dit_type: str,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            *args
    ):
        "Forward method with activation checkpointing."
        if dit_type == "double_stream":
            recompute_layers = self.num_layers
        elif dit_type == "single_stream":
            recompute_layers = self.num_single_layers
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
        
        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = self._get_block(dit_type, index)
                    x_ = layer(*args)
                return x_
            return custom_forward
        
        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < self.num_layers:
                hidden_states, encoder_hidden_states = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.config.distribute_saved_activations,
                    hidden_states,
                    encoder_hidden_states,
                    *args
                )
                _layer_num += recompute_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.

            for _layer_num in range(recompute_layers):
                if _layer_num < recompute_layers:
                    hidden_states, encoder_hidden_states = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.config.distribute_saved_activations,
                        hidden_states,
                        encoder_hidden_states,
                        *args
                    )
                else:
                    block = self._get_block(dit_type, _layer_num)
                    hidden_states, encoder_hidden_states = block(*hidden_states, *encoder_hidden_states, *args)

        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")
        
        return  hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.patch_size, self.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        freqs_cos, freqs_sin = self.rope(hidden_states)
        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        # 3. Attention mask preparation
        if encoder_attention_mask is None:
            attention_mask = None
        else:
            latent_sequence_length = hidden_states.shape[1]
            condition_sequence_length = encoder_hidden_states.shape[1]
            sequence_length = latent_sequence_length + condition_sequence_length
            attention_mask = torch.zeros(
                batch_size,
                sequence_length,
                sequence_length,
                device=hidden_states.device,
                dtype=torch.bool,
            )  # [B, N, N]

            effective_condition_sequence_length = encoder_attention_mask.sum(
                dim=1, dtype=torch.int
            )  # [B,]
            effective_sequence_length = (
                latent_sequence_length + effective_condition_sequence_length
            )

            for i in range(batch_size):
                attention_mask[
                    i, : effective_sequence_length[i], : effective_sequence_length[i]
                ] = True

        hidden_states=hidden_states.contiguous()
        encoder_hidden_states=encoder_hidden_states.contiguous()
        temb=temb.contiguous()

        if mpu.get_context_parallel_world_size() > 1:
            length = hidden_states.shape[1]
            set_origin_length(length)
            seq_parallel_world_size = mpu.get_context_parallel_world_size()
            if length % seq_parallel_world_size != 0:
                pad_size = seq_parallel_world_size - (length % seq_parallel_world_size)
                length = length + pad_size
            set_target_length(length)
            hidden_states = pad_for_context_parallel(hidden_states, 1)
            freqs_cos = pad_for_context_parallel(freqs_cos, 0)
            freqs_sin = pad_for_context_parallel(freqs_sin, 0)
            
            hidden_states = split_forward_gather_backward(
                hidden_states, 
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            ) # b s n ds
            freqs_cos = split_forward_gather_backward(
                freqs_cos,
                mpu.get_context_parallel_group(),
                dim=0,
                grad_scale="down"
            )
            freqs_sin = split_forward_gather_backward(
                freqs_sin,
                mpu.get_context_parallel_group(),
                dim=0,
                grad_scale="down"
            )
        # 4. Transformer blocks
        if self.config.recompute_granularity == "full":
            hidden_states, encoder_hidden_states = self._checkpointed_forward(
                "double_stream",
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                freqs_cos,
                freqs_sin,
            )

            hidden_states, encoder_hidden_states = self._checkpointed_forward(
                "single_stream",
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                freqs_cos,
                freqs_sin,
            )
        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    freqs_cos,
                    freqs_sin,
                )
            if mpu.get_context_parallel_world_size() > 1:
                hidden_states = remove_pad_for_context_parallel(hidden_states, 1)
                freqs_cos = remove_pad_for_context_parallel(freqs_cos, 0)
                freqs_sin = remove_pad_for_context_parallel(freqs_sin, 0)

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    freqs_cos,
                    freqs_sin,
                )

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states, 
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="up"
            )
            hidden_states = remove_pad_for_context_parallel(hidden_states, 1)
        # 5. Output projection

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
