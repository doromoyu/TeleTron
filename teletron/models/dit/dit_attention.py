# Copyright 2025 TeleAI-infra Team and HuggingFace Inc. All rights reserved.

from dataclasses import dataclass
from typing import Union
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
import torch
from megatron.core import mpu
from diffusers.models.embeddings import apply_rotary_emb
from megatron.core.transformer.attention import Attention, SelfAttention
from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from teletron.core.tensor_parallel.mappings import split_forward_gather_backward, gather_forward_split_backward 
from teletron.core.context_parallel.pad import pad_for_context_parallel, remove_pad_for_context_parallel, remove_pad_with_encoder_for_context_parallel
import torch.nn.functional as F


@dataclass
class JointSelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    added_linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    added_q_layernorm: Union[ModuleSpec, type] = None
    added_k_layernorm: Union[ModuleSpec, type] = None


class JointHunyuanAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    added_linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    added_q_layernorm: Union[ModuleSpec, type] = None
    added_k_layernorm: Union[ModuleSpec, type] = None
    attn_bda: Union[ModuleSpec, type] = IdentityFuncOp
    

class JointHunyuanAttention(Attention):
    """Joint Self-attention layer class

    Used for MMDIT-like transformer block.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: JointHunyuanAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        context_pre_only: bool = False,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self"
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.added_linear_qkv is not None:
            self.added_linear_qkv = build_module(
                submodules.added_linear_qkv,
                self.config.hidden_size,
                self.query_projection_size + 2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='qkv',
            )

        if not context_pre_only:
            self.added_linear_proj = build_module(
                submodules.linear_proj,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=self.config.add_bias_linear,
                input_is_parallel=True,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name='proj',
            )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None
        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        if submodules.added_q_layernorm is not None:
            self.added_q_layernorm = build_module(
                submodules.added_q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_q_layernorm = None

        if submodules.added_k_layernorm is not None:
            self.added_k_layernorm = build_module(
                submodules.added_k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.added_k_layernorm = None
        if hasattr(submodules, 'attn_bda') and submodules.attn_bda is not None:
            self.attn_bda = build_module(submodules.attn_bda)
        else:
            self.attn_bda = None

    def _split_qkv(self, mixed_qkv):
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (    # [1, 360] + (12, ())
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        
        query, key, value = self._split_qkv(mixed_qkv)  # [2, 9604, 12, 128] [b, s, num_heads, hiddensize_per_head]


        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def get_added_query_key_value_tensors(self, added_hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.added_linear_qkv(added_hidden_states)

        query, key, value = self._split_qkv(mixed_qkv)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.added_q_layernorm is not None:
            query = self.added_q_layernorm(query)

        if self.added_k_layernorm is not None:
            key = self.added_k_layernorm(key)

        return query, key, value
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        freqs_cos=None,
        freqs_sin=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        additional_hidden_states=None,
    ):
        # hidden_states: [sq, b, h]

        query, key, value = self.get_query_key_value_tensors(hidden_states) # (b,h,s,d)
        added_query, added_key, added_value = self.get_added_query_key_value_tensors(additional_hidden_states)
        bs, _, img_seq_len, _ = query.shape
        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if freqs_cos!=None and freqs_sin!=None:
            rotary_pos_emb=(freqs_cos,freqs_sin)
        else:
            rotary_pos_emb=None

        if rotary_pos_emb is not None:
            query = apply_rotary_emb(
                query,
                rotary_pos_emb
            )
            key = apply_rotary_emb(
                key,
                rotary_pos_emb
            )
        
        if mpu.get_context_parallel_world_size() > 1:
            from yunchang.comm.all_to_all import SeqAllToAll4D
            query = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), query, 1, 2)
            key = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),key,  1, 2)
            value = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),value,  1, 2)
            
            torch.cuda.empty_cache() # call empty cache, otherwise qkv's will take extra GPU memory
            added_query = split_forward_gather_backward(added_query,mpu.get_context_parallel_group(), dim=1)
            added_key = split_forward_gather_backward(added_key,mpu.get_context_parallel_group(), dim=1)
            added_value= split_forward_gather_backward(added_value,mpu.get_context_parallel_group(), dim=1)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            
            query, key ,value = map(
                lambda x: remove_pad_for_context_parallel(x, dim=2),
                [query, key, value]
            )
        encoder_length = added_query.shape[2]

        query = torch.cat([query, added_query], dim=2).permute(2, 0, 1, 3).contiguous()  # bhsd -> sbhd
        key = torch.cat([key, added_key], dim=2).permute(2, 0, 1, 3).contiguous()
        value = torch.cat([value, added_value], dim=2).permute(2, 0, 1, 3).contiguous()

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
            )
        else:
            query, key, value = [x.permute(1, 2, 0, 3).contiguous() for x in (query, key, value)] # sbhd -> bhsd
            torch.backends.cuda.enable_cudnn_sdp(False)
            core_attn_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = core_attn_out[:, :, : -encoder_length, :]
        encoder_hidden_states = core_attn_out[:, :, -encoder_length : , :]

        hidden_states = hidden_states.transpose(1, 2)   # bnsd -> bsnd
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)   # bnsd -> bsnd

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = pad_for_context_parallel(hidden_states, 1)

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), hidden_states, 1, 2) # b img_seq sub_n d
            torch.cuda.empty_cache() # call empty cache, otherwise qkv's will take extra GPU memory
            encoder_hidden_states = gather_forward_split_backward(encoder_hidden_states, mpu.get_context_parallel_group(), 2) # b txt_seq n d

        hidden_states = hidden_states.flatten(2, 3).contiguous()
        encoder_hidden_states = encoder_hidden_states.flatten(2, 3).contiguous()

        output, bias = self.linear_proj(hidden_states)
        encoder_output, encoder_bias = self.added_linear_proj(encoder_hidden_states)

        output = output + bias
        encoder_output = encoder_output + encoder_bias

        return output, encoder_output

class HunyuanSingleAttention(SelfAttention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def _split_qkv(self, mixed_qkv):
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(
                mixed_qkv,
                3,
                split_arg_list,
            )
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(
                mixed_qkv,
                split_arg_list,
                dim=3,
            )

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        return query, key, value

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        query, key, value = self._split_qkv(mixed_qkv)  # [2, 9604, 12, 128] [b, s, num_heads, hiddensize_per_head]sss

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.config.test_mode:
            self.run_realtime_tests()

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        return query, key, value

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        freqs_cos=None,
        freqs_sin=None,
        packed_seq_params=None,
        additional_hidden_states=None,
    ):
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        hidden_states = torch.cat([hidden_states, additional_hidden_states], dim=1)
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        if freqs_cos!=None and freqs_sin!=None:
            rotary_pos_emb=(freqs_cos,freqs_sin)
        else:
            rotary_pos_emb=None
        img_query = apply_rotary_emb(
                    query[:, :, : -additional_hidden_states.shape[1]],
                    rotary_pos_emb,
                )
        txt_query = query[:, :, -additional_hidden_states.shape[1] :]
        img_key = apply_rotary_emb(
                    key[:, :, : -additional_hidden_states.shape[1]],
                    rotary_pos_emb,
                )
        txt_key=key[:, :, -additional_hidden_states.shape[1] :]
        image_value = value[:, :, : -additional_hidden_states.shape[1]]
        txt_value = value[:, :, -additional_hidden_states.shape[1] :]

        if mpu.get_context_parallel_group is not None:
            from yunchang.comm.all_to_all import SeqAllToAll4D
            query = SeqAllToAll4D.apply(mpu.get_context_parallel_group(), img_query,  1, 2)
            key = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),img_key, 1, 2)
            value = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),image_value, 1, 2)

            added_query = split_forward_gather_backward(txt_query,mpu.get_context_parallel_group(), dim=1)
            added_key = split_forward_gather_backward(txt_key,mpu.get_context_parallel_group(), dim=1)
            added_value= split_forward_gather_backward(txt_value,mpu.get_context_parallel_group(), dim=1)
            del img_query,img_key,image_value,txt_query,txt_key,txt_value
            torch.cuda.empty_cache() # call empty cache, otherwise qkv's will take extra GPU memory
        query = torch.cat([query, added_query], dim=2).permute(2, 0, 1, 3).contiguous()  # bhsd -> sbhd
        key = torch.cat([key, added_key], dim=2).permute(2, 0, 1, 3).contiguous()
        value = torch.cat([value, added_value], dim=2).permute(2, 0, 1, 3).contiguous()
        
        encoder_length = added_query.shape[2]

        del added_query,added_key,added_value

        if mpu.get_context_parallel_world_size() > 1:
            query, key ,value = map(
                lambda x: remove_pad_with_encoder_for_context_parallel(x, encoder_length, dim=0),
                [query, key, value]
            )

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
            )
        else:
            import torch.nn.functional as F
            query, key, value = [x.permute(1, 2, 0, 3).contiguous() for x in (query, key, value)] # sbhd -> bhsd
            torch.backends.cuda.enable_cudnn_sdp(False)
            core_attn_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        core_attn_out=core_attn_out.to(query.dtype)
        hidden_states = core_attn_out[:, :, :-additional_hidden_states.shape[1], :]
        encoder_hidden_states = core_attn_out[:, :,-additional_hidden_states.shape[1]:, :]

        hidden_states = hidden_states.transpose(1, 2)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = pad_for_context_parallel(hidden_states, 1)

            hidden_states = SeqAllToAll4D.apply(mpu.get_context_parallel_group(),hidden_states, 1, 2) # b img_seq sub_n d
            encoder_hidden_states = gather_forward_split_backward(encoder_hidden_states, mpu.get_context_parallel_group(), 2) # b txt_seq n d
            torch.cuda.empty_cache() # call empty cache, otherwise qkv's will take extra GPU memory

        hidden_states = hidden_states.flatten(2, 3).contiguous()
        encoder_hidden_states = encoder_hidden_states.flatten(2, 3).contiguous()

        return hidden_states, encoder_hidden_states