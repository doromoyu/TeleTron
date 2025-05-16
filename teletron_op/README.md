# Installation
```
cd teletron_op && bash install.sh
```
# Usage
- Enable cuda fused kernels with `export FUSED_KERNELS=1`

# Operator List

## Fused AdaLayerNorm 

Currently it only supports the last dimension size of the input tensor to be 3072
```python
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
```

## Fused RMSNorm 
Currently only supports the last dimension size of the input tensor to be 128
```python
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
```