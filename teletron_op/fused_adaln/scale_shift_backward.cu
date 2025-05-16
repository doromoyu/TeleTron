// Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
#include <cuda_fp16.h> 
#include <ATen/ATen.h>

template<int blockSize=1024>
__device__ inline void blockReduceSum(float val, float* shared_temp) {
    int tid = threadIdx.y;
    int block_size = blockDim.y;
    int warp_size = 32;
    int lane_id = tid % warp_size;
    int warp_id = tid / warp_size;
    constexpr int warp_num = blockSize >> 5;
    static __shared__ float warpSum[warp_num];


    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }


    if (lane_id == 0) {
        warpSum[warp_id] = val;
    }


    __syncthreads();


    val = (tid < warp_num) ? warpSum[tid] : 0.0f;

    if (warp_id == 0) {
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
             val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    
    if (tid == 0) {
        shared_temp[0] = val;
    }
}

template<typename ComputeType>
__global__ void LayerNormBlockSMemImpl_weight_shift_bp_opt(
    ComputeType* dscale, ComputeType* dshift, 
    const ComputeType* dout, const ComputeType* x_norm, 
    const int64_t rows, const int64_t cols
);

template<>
__global__ void LayerNormBlockSMemImpl_weight_shift_bp_opt<__nv_bfloat16>(
    __nv_bfloat16* dscale, __nv_bfloat16* dshift, 
    const __nv_bfloat16* dout, const __nv_bfloat16* x_norm, 
    const int64_t rows, const int64_t cols
) {
    constexpr int pack_size = 8; 
    constexpr int final_sum_size = pack_size * 2;
    __shared__ float shared_mem[16];

    float* final_sums = shared_mem; 
    const int tid = threadIdx.y;
    const int block_size = blockDim.y;
    const int col_id_start = blockIdx.x * pack_size;

    using LoadStoreType = float4;

    float thread_dscale[pack_size] = {0.f};
    float thread_dshift[pack_size] = {0.f};

    for (int64_t row = tid; row < rows; row += block_size) {
        const int64_t row_offset = row * cols;
        auto current_dout_packed = *reinterpret_cast<const LoadStoreType*>(dout + row_offset + col_id_start);
        auto current_x_norm_packed = *reinterpret_cast<const LoadStoreType*>(x_norm + row_offset + col_id_start);

        float2 d01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_dout_packed.x));
        float2 d23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_dout_packed.y));
        float2 d45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_dout_packed.z));
        float2 d67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_dout_packed.w));

        float2 xn01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_x_norm_packed.x));
        float2 xn23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_x_norm_packed.y));
        float2 xn45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_x_norm_packed.z));
        float2 xn67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&current_x_norm_packed.w));

        thread_dshift[0] += d01.x; thread_dshift[1] += d01.y;
        thread_dshift[2] += d23.x; thread_dshift[3] += d23.y;
        thread_dshift[4] += d45.x; thread_dshift[5] += d45.y;
        thread_dshift[6] += d67.x; thread_dshift[7] += d67.y;

        thread_dscale[0] += d01.x * xn01.x; thread_dscale[1] += d01.y * xn01.y;
        thread_dscale[2] += d23.x * xn23.x; thread_dscale[3] += d23.y * xn23.y;
        thread_dscale[4] += d45.x * xn45.x; thread_dscale[5] += d45.y * xn45.y;
        thread_dscale[6] += d67.x * xn67.x; thread_dscale[7] += d67.y * xn67.y;
    }

    if (tid < final_sum_size) {
        final_sums[tid] = 0.0f;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < pack_size; ++i) {

        blockReduceSum(thread_dscale[i], final_sums+i);

        blockReduceSum(thread_dshift[i],  final_sums+i + pack_size);
    }

    if (tid == 0) {

        __nv_bfloat162 final_dscale_packed[4];
        __nv_bfloat162 final_dshift_packed[4];

        final_dscale_packed[0] = (__floats2bfloat162_rn(final_sums[0], final_sums[1]));
        final_dscale_packed[1] = (__floats2bfloat162_rn(final_sums[2], final_sums[3]));
        final_dscale_packed[2] = (__floats2bfloat162_rn(final_sums[4], final_sums[5]));
        final_dscale_packed[3] = (__floats2bfloat162_rn(final_sums[6], final_sums[7]));

        final_dshift_packed[0] = (__floats2bfloat162_rn(final_sums[8], final_sums[9]));
        final_dshift_packed[1] = (__floats2bfloat162_rn(final_sums[10], final_sums[11]));
        final_dshift_packed[2] = (__floats2bfloat162_rn(final_sums[12], final_sums[13]));
        final_dshift_packed[3] = (__floats2bfloat162_rn(final_sums[14], final_sums[15]));

        *reinterpret_cast<LoadStoreType*>(dscale + col_id_start) = *reinterpret_cast<LoadStoreType*>(final_dscale_packed);
        *reinterpret_cast<LoadStoreType*>(dshift + col_id_start) = *reinterpret_cast<LoadStoreType*>(final_dshift_packed);
    }
}


void launch_adaln_scale_shift_backward(
    __nv_bfloat16* d_dscale, __nv_bfloat16* d_dshift,
    const __nv_bfloat16* d_dout, const __nv_bfloat16* d_x_norm, 
    const int64_t N, const int64_t C)
{

    auto row_grid = (N > 1024)? 1024:(N+ 31) / 32 * 32;
    dim3 blockDim(1, row_grid, 1); 

    dim3 gridDim(C / 8, 1, 1);
    LayerNormBlockSMemImpl_weight_shift_bp_opt<__nv_bfloat16><<<gridDim, blockDim>>>(
        d_dscale,
        d_dshift,
        d_dout,
        d_x_norm,
        N, 
        C 
    );
}
