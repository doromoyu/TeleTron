// Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
#include <torch/extension.h>
#include "adaln_kernel.h"


void torch_launch_adaln_forward(
                                        torch::Tensor &output, 
                                        torch::Tensor &x_norm,
                                       const torch::Tensor &input,
                                       const torch::Tensor &scale,
                                       const torch::Tensor &shift,
                                       int64_t rows,
                                       int64_t cols,
                                       double epsilon,
                                       torch::Tensor &inv_variance) {
    launch_adaln_forward(
        (at::BFloat16 *)(output.data_ptr()),
        (at::BFloat16 *)(x_norm.data_ptr()),
        (const at::BFloat16 *)(input.data_ptr()),
        (const at::BFloat16 *)(scale.data_ptr()),
        (const at::BFloat16 *)(shift.data_ptr()),
        rows,
        cols,
        epsilon,
        (float *)(inv_variance.data_ptr())
    );
}

void torch_launch_adaln_backward(
                                        torch::Tensor &dinp,
                                        torch::Tensor &dscale,
                                        torch::Tensor &dshift,
                                        const torch::Tensor &dout,
                                        const torch::Tensor &x_norm,
                                        const torch::Tensor &scale,
                                        const torch::Tensor &inv_variance,
                                        int64_t rows,
                                        int64_t cols) {
    launch_adaln_backward( 
        (at::BFloat16 *)(dinp.data_ptr()),
        (at::BFloat16 *)(dscale.data_ptr()),
       (at::BFloat16 *)(dshift.data_ptr()),
        (const at::BFloat16 *)(dout.data_ptr()),
        (const at::BFloat16 *)(x_norm.data_ptr()),
        (const at::BFloat16 *)(scale.data_ptr()),
        static_cast<const float*>(inv_variance.data_ptr()),
        rows,
        cols 
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_adaln_forward",
          &torch_launch_adaln_forward,
          "AdaLN forward kernel");
    
    m.def("torch_launch_adaln_backward",
          &torch_launch_adaln_backward,
          "AdaLN backward kernel");
}

TORCH_LIBRARY(layernorm_block_smem, m) {
    m.def("torch_launch_adaln_forward", torch_launch_adaln_forward);
    m.def("torch_launch_adaln_backward", torch_launch_adaln_backward);
}