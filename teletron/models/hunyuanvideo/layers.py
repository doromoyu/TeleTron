# Copyright 2025 TeleAI-infra Team and HuggingFace Inc. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)

class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states
    
class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            dropout=mlp_drop_rate,
        )

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: List[int],
        theta: float = 256.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(
                0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32
            )
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(
                self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True
            )
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin