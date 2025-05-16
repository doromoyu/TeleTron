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
#include <ATen/cuda/CUDAContext.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <adaln_kernel.h>

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;
}

template<typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 16; mask > 0; mask >>=1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T, int blockSize = 256>
__device__ void blockReduceSum(T val, T* dout){
    int tid = threadIdx.x;
    int laneid = tid % 32;
    int warpid = tid / 32;
    constexpr int warp_num = blockSize >> 5;

    static __shared__ T warpSum[warp_num];

    val = warpReduceSum<T>(val);

    if(laneid == 0) warpSum[warpid] = val; 
    __syncthreads();

    T sum = (tid < warp_num)? warpSum[tid]:0;
    sum = warpReduceSum<T>(sum);
    if (tid == 0)
    {
        *dout = sum;
    }
}

template<const int block_size=384>
__global__ void AdaLNImpl_bp(
    __nv_bfloat16* dinp, 
    const __nv_bfloat16* dout, 
    const __nv_bfloat16* x_norm, 
    const __nv_bfloat16* scale, 
    const float* inv_variance,
    const int64_t rows, const int64_t cols
) {
    __shared__ __nv_bfloat16 buf[3072];
    constexpr int pack_size = 8;
    const int tid = threadIdx.x;
    const int num_packs_per_row = cols / pack_size;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x)
    {
        const int row_offset = row * cols;
        const __nv_bfloat16* cur_dout_row = dout + row_offset;
        const __nv_bfloat16* cur_x_norm_row = x_norm + row_offset;
        const __nv_bfloat16* cur_scale_row = scale ;
        __nv_bfloat16* cur_dinp_row = dinp + row_offset;
        const float invvar = inv_variance[row];

        float sdout_m = 0.0f;
        float sdout_norm_m = 0.0f;

        for (int pack_id = tid; pack_id < num_packs_per_row; pack_id += block_size) {
            const int col_offset = pack_id * pack_size;
            auto scale_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(cur_scale_row + col_offset);
            __nv_bfloat162 pack_dout[pack_size / 2];
            __nv_bfloat162 pack_x_norm[pack_size / 2];
            *reinterpret_cast<float4*>(pack_dout) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(cur_dout_row +col_offset));
            *reinterpret_cast<float4*>(pack_x_norm) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(cur_x_norm_row + col_offset));

            #pragma unroll
            for (int i = 0; i < pack_size/2; ++i) {
                auto sdout_i = __hfma2(scale_bf162_ptr[i] , pack_dout[i], pack_dout[i]);
                reinterpret_cast<__nv_bfloat162*>(buf)[i * num_packs_per_row + pack_id] = sdout_i;
                sdout_m += __bfloat162float(sdout_i.x + sdout_i.y);

                auto sdout_norm_i = __bfloat1622float2(__hmul2(sdout_i, pack_x_norm[i]));
                sdout_norm_m += sdout_norm_i.x + sdout_norm_i.y;
            }
        }
        __shared__ float sm_norm[2];
        blockReduceSum<float, block_size>(sdout_m, sm_norm);
        blockReduceSum<float, block_size>(sdout_norm_m, sm_norm+1);
        __syncthreads();
        sdout_m = Div<float>(sm_norm[0], cols);
        sdout_norm_m = Div<float>(sm_norm[1], cols);

        for (int pack_id = tid; pack_id < num_packs_per_row; pack_id += block_size) {
            const int col_offset = pack_id * pack_size;
            __nv_bfloat162 pack[pack_size / 2];
            *reinterpret_cast<float4*>(pack) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(cur_x_norm_row + col_offset));
            #pragma unroll
            for (int i = 0; i < pack_size / 2; ++i) {
                auto val = __bfloat1622float2(reinterpret_cast<__nv_bfloat162*>(buf)[i * num_packs_per_row + pack_id]);
                auto x_norm_f2 = __bfloat1622float2(pack[i]);
                pack[i] = __floats2bfloat162_rn(invvar * (val.x - sdout_m - x_norm_f2.x * sdout_norm_m ), 
                                                                                    invvar * (val.y - sdout_m - x_norm_f2.y * sdout_norm_m ));
            }
            *reinterpret_cast<float4*>(cur_dinp_row + col_offset) = *reinterpret_cast<float4*>(pack);
        }
    }
}

void launch_adaln_backward(
    at::BFloat16* dinp, 
    at::BFloat16* dscale,
    at::BFloat16* dshift, 
    const at::BFloat16* dout, 
    const at::BFloat16* x_norm, const at::BFloat16* scale, const float* inv_variance,
    const int64_t rows, const int64_t cols
) {
    launch_adaln_scale_shift_backward(
    (__nv_bfloat16*)dscale,(__nv_bfloat16*)dshift,      
    ( const __nv_bfloat16*)dout, ( const __nv_bfloat16*)x_norm,
    rows, cols              
    );
    auto row_grid = (rows <1200 )? rows:rows>>2;
    dim3 grid(row_grid);
    dim3 block(384);
    AdaLNImpl_bp<384><<<grid, block>>>(
    (__nv_bfloat16*)dinp, 
    ( const __nv_bfloat16*)dout, 
    ( const __nv_bfloat16*)x_norm,  ( const __nv_bfloat16*)scale,  inv_variance,
    rows, cols
    );
}