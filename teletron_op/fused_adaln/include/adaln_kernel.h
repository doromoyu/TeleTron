/*
 * Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

void launch_adaln_forward(
	at::BFloat16 *  output,     
    at::BFloat16 * x_norm,
    const at::BFloat16 *  input, 
    const at::BFloat16 *  scale, 
    const at::BFloat16 *  shift, 
    const int64_t rows,         
    const int64_t cols,         
    const double epsilon,       
    float* inv_variance);

void launch_adaln_scale_shift_backward(
    __nv_bfloat16* d_dscale, __nv_bfloat16* d_dshift,      
    const __nv_bfloat16* d_dout, const __nv_bfloat16* d_x_norm,
    const int64_t N, const int64_t C);

void launch_adaln_backward(
    at::BFloat16* dinp, 
    at::BFloat16* dscale,
    at::BFloat16* dshift, 
    const at::BFloat16* dout, 
    const at::BFloat16* x_norm, 
    const at::BFloat16* scale, 
    const float* inv_variance,
    const int64_t rows, const int64_t cols
);