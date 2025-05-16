# Copyright 2025 TeleAI-infra Team and HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Function
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

import os
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

try:
    import fused_adaln
except ImportError:
    log.warning("fused_adaln module not imported. Some features might be unavailable.")
    fused_adaln = None

try:
    import fused_rmsnorm
except ImportError:
    log.warning("fused_rmsnorm module not imported. Some features might be unavailable.")
    fused_rmsnorm = None


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)

class AdaLNModelFunction(Function):
    """
    Implements custom AdaLN operation as a PyTorch autograd Function.

    Currently supports specific tensor shape with cols=3072 for optimized performance.
    """
    @staticmethod
    def forward(ctx, x, scale, shift, epsilon, cols):
        """
        Forward pass computation for AdaLN.
        
        Args:
            ctx: Context object for saving tensors needed in backward pass
            x: Input tensor
            scale: Scaling parameter of shape [cols]
            shift: Shifting parameter of shape [cols]
            epsilon: Small constant for numerical stability
            cols: Feature dimension size (must be 3072 in current implementation)
        
        Returns:
            output: Normalized output tensor with same shape as input x
            
        Note:
            Temporarily only supports cols=3072 due to CUDA kernel optimization constraints.
            Input tensor size must be divisible by 3072.
        """
        x = x.contiguous()
        scale = scale.contiguous()
        shift_= shift.contiguous()

        ctx.cols = cols
        ctx.rows = x.numel() // cols
        if x.numel() % cols != 0 or cols != 3072:
            raise ValueError(f"Input tensor size {x.numel()} not divisible by cols {cols}")
        ctx.eps = epsilon

        output = torch.empty_like(x)
        x_norm = torch.empty_like(x)

        invvar = torch.empty(ctx.rows, device=x.device, dtype=torch.float32)
        # Launch CUDA kernel through custom extension
        fused_adaln.torch_launch_adaln_forward(
            output, x_norm, x, scale, shift_, ctx.rows, ctx.cols, ctx.eps, invvar
        )
        # Save the intermediate variables for back propagation
        ctx.save_for_backward(x_norm, scale, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computation for AdaLN.
        
        Args:
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient of the output tensor
            
        Returns:
            grad_input: Gradient for input x
            grad_scale: Gradient for scale
            grad_shift: Gradients for shift
            None: Placeholder for epsilon gradient
            None: Placeholder for cols gradient
        
        """
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        x_norm, scale_, invvar = ctx.saved_tensors
        grad_input = torch.empty_like(x_norm)
        grad_scale = torch.empty_like(scale_)
        grad_shift = torch.empty_like(scale_) 

        fused_adaln.torch_launch_adaln_backward(
            grad_input, grad_scale,grad_shift, 
            grad_output,
            x_norm, scale_, invvar, ctx.rows, ctx.cols
        )
        return grad_input, grad_scale, grad_shift, None, None

class AdaLNCustom(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x, scale, shift):
        cols = self.hidden_size
        if x.shape[-1] != cols:
             raise ValueError(f"Last dim of input x must be hidden_size {cols}, got {x.shape[-1]}")
        if scale.shape[-1] != cols:
             raise ValueError(f"Last dim of scale must be hidden_size {cols}, got {scale.shape[-1]}")
        if shift.shape[-1] != cols:
             raise ValueError(f"Last dim of shift must be hidden_size {cols}, got {shift.shape[-1]}")
        return AdaLNModelFunction.apply(x, scale, shift, self.eps, cols)

class FusedAdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, 
                 num_embeddings: Optional[int] = None, 
                 norm_type="layer_norm", 
                 bias=True, 
                 fused_kernels = False,
                 linear_parallel=True,
                 config=None):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        
        self.linear_parallel = linear_parallel
        if not linear_parallel:
            self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        else:
            assert config is not None
            self.linear = TEColumnParallelLinear(
                    input_size=embedding_dim,  
                    output_size=6 * embedding_dim,
                    config=config,
                    gather_output=False,
                    init_method=config.init_method,
                    bias=bias,
                    skip_bias_add=False,
                    is_expert=False)
        self.fused_kernels = fused_kernels
        if fused_kernels and fused_adaln and embedding_dim == 3072:
            self.adaLN = AdaLNCustom(embedding_dim,  eps=1e-6)
        else:
            if norm_type == "layer_norm":
                self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
            elif norm_type == "fp32_layer_norm":
                self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
            else:
                raise ValueError(
                    f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
                )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        if self.linear_parallel:
            emb = gather_from_tensor_model_parallel_region(emb[0])
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        if self.fused_kernels and fused_adaln:
            x = self.adaLN(x, scale_msa, shift_msa)
        else:
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class FusedAdaLayerNormZeroSingle(nn.Module):

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True, fused_kernels = False, linear_parallel=True, config=None):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear_parallel = linear_parallel
        if not linear_parallel:
            self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        else:
            assert config is not None
            self.linear = TEColumnParallelLinear(
                    input_size=embedding_dim,  
                    output_size=3 * embedding_dim,
                    config=config,
                    gather_output=False,
                    init_method=config.init_method,
                    bias=bias,
                    skip_bias_add=False,
                    is_expert=False)
        self.fused_kernels = fused_kernels
        if fused_kernels and fused_adaln and embedding_dim == 3072:
            self.adaLN = AdaLNCustom(embedding_dim,  eps=1e-6)
        else:
            if norm_type == "layer_norm":
                self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
            elif norm_type == "fp32_layer_norm":
                self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
            else:
                raise ValueError(
                    f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
                )

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        if self.linear_parallel:
            emb = gather_from_tensor_model_parallel_region(emb[0])
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        if self.fused_kernels and fused_adaln:
            x = self.adaLN(x, scale_msa, shift_msa)
        else:
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa

class RMSNormModelFunction(Function):
    """
    Implements custom RMSNorm operation as a PyTorch autograd Function.
    Uses fused CUDA kernels for accelerated computation. Currently optimized for cols=128 only.
    """
    
    @staticmethod
    def forward(ctx, x, weight, epsilon, cols):
        """
        Forward pass for RMSNorm operation.
        
        Args:
            ctx: Context object for saving tensors needed in backward pass
            x: Input tensor of shape
            weight: Learnable scaling parameter of shape [cols]
            epsilon: Small constant for numerical stability (1e-6 typical)
            cols: Feature dimension size (must be 128 in current implementation)
            
        Returns:
            output: Normalized tensor with same shape as input x

        """
        x = x.contiguous()
        weight = weight.contiguous()

        ctx.cols = cols
        ctx.rows = x.numel() // cols
        if x.numel() % cols != 0 or cols != 128:
            raise ValueError(f"Input size {x.numel()} not divisible by cols {cols} or unsupported cols value (must be 128)")
        ctx.eps = epsilon

        output = torch.empty_like(x)
        invvar = torch.empty(ctx.rows, device=x.device, dtype=torch.float32) 

        # Launch optimized CUDA kernel
        fused_rmsnorm.torch_launch_rms_forward(
            output, x, weight, ctx.rows, ctx.cols, ctx.eps, invvar
        )

        ctx.save_for_backward(output, weight, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for RMSNorm operation.
        
        Args:
            ctx: Context containing saved tensors from forward pass
            grad_output: Gradient of loss 
            
        Returns:
            grad_input: Gradient for input x
            grad_weight: Gradient for weight parameter
            None: Placeholder for epsilon gradient
            None: Placeholder for cols gradient
        
        """
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        fwd_pass_output, weight, invvar = ctx.saved_tensors

        grad_input = torch.empty_like(fwd_pass_output)  
        grad_weight = torch.empty_like(weight)          

        # Launch optimized CUDA backward kernel
        fused_rmsnorm.torch_launch_rms_backward(
            grad_input, grad_weight,
            grad_output, 
            fwd_pass_output,
            weight, invvar, ctx.rows, ctx.cols
        )

        return grad_input, grad_weight, None, None

def Get_RMSNorm():
    _RMSNormImplementation = None

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, config, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.hidden_size = hidden_size

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            if os.environ.get("FUSED_KERNELS"):
                fused_kernels_bool = bool(int(os.environ.get("FUSED_KERNELS")))
                if fused_kernels_bool is True and fused_rmsnorm is not None and self.hidden_size == 128:
                    return RMSNormModelFunction.apply(x, self.weight, self.eps, self.hidden_size)

            output = self._norm(x.float()).type_as(x)
            return output * self.weight

    if _RMSNormImplementation is None:
        _RMSNormImplementation = RMSNorm
    
    return _RMSNormImplementation


