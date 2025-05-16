#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <rmsnorm_kernel.h>

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;
}

template<typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 8; mask > 0; mask >>=1)
    {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T>
__device__ void blockReduceSum(T val, T* dout){
    int tidx = threadIdx.x;
    val = warpReduceSum<T>(val);

    if(tidx == 0) dout[threadIdx.y] = val;
    __syncthreads();
}

template<const int block_size=256>
__global__ void RMSNormImpl_bp(
    __nv_bfloat16* dinp, 
    const __nv_bfloat16* dout, 
    const __nv_bfloat16* x_norm, 
    const __nv_bfloat16* weight, 
    const float* inv_variance,
    const int64_t rows, const int64_t cols
) {
    __shared__ float sum_sm[16];

    constexpr int pack_size = 8;
    const int tidx = threadIdx.x;  
    const int tidy = threadIdx.y;  

    for(int64_t row = blockIdx.y * blockDim.y + tidy; row < rows; row += gridDim.y * blockDim.y) {
        const __nv_bfloat16* cur_dout_row = dout + row * cols;
        const __nv_bfloat16* cur_x_norm_row = x_norm + row * cols;
        __nv_bfloat16* cur_dinp_row = dinp + row * cols;
        const float invvar = inv_variance[row];

        float x2_pack_sum = 0.0f;
        const int col_offset = tidx * pack_size;
        __nv_bfloat162 pack_weight[pack_size/2];
        __nv_bfloat162 pack_dout[pack_size / 2];
        __nv_bfloat162 pack_x_norm[pack_size / 2];
        *reinterpret_cast<float4*>(pack_dout) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(cur_dout_row +col_offset));
        *reinterpret_cast<float4*>(pack_x_norm) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(cur_x_norm_row + col_offset));
        *reinterpret_cast<float4*>(pack_weight) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(weight + col_offset));
        

        #pragma unroll
        for(int i = 0; i < pack_size/2; ++i) {
            auto doutWX_i = __bfloat1622float2(__hmul2(pack_dout[i], pack_x_norm[i]));
            
            x2_pack_sum += doutWX_i.x   + doutWX_i.y;
        }
        blockReduceSum(x2_pack_sum, sum_sm);

        x2_pack_sum = Div<float>(sum_sm[tidy], cols);

       
        #pragma unroll
        for (int i = 0; i < pack_size / 2; ++i) {

            auto d2 = __bfloat1622float2(pack_dout[i]);
            auto w2 = __bfloat1622float2(pack_weight[i]);
            auto x2 = __bfloat1622float2(pack_x_norm[i]);

            auto dw1 = d2.x * w2.x; auto dw2 = d2.y * w2.y;
            auto rsw1 = x2.x / w2.x; auto rsw2 = x2.y / w2.y; 

            pack_dout[i] = __floats2bfloat162_rn(invvar *( dw1 - x2_pack_sum * rsw1 ), invvar * (dw2 - x2_pack_sum * rsw2));
        }
        *reinterpret_cast<float4*>(cur_dinp_row + col_offset) = *reinterpret_cast<float4*>(pack_dout);
    }
}

void launch_rms_backward(
    at::BFloat16* dinp, 
    at::BFloat16* dweight,
    const at::BFloat16* dout, 
    const at::BFloat16* x_rmsnorm, const at::BFloat16* weight, const float* inv_variance,
    const int64_t rows, const int64_t cols
) {
    launch_rms_weight_backward(
        (__nv_bfloat16*)dweight,  
        ( const __nv_bfloat16*)dout, ( const __nv_bfloat16*)x_rmsnorm, ( const __nv_bfloat16*)weight,
        rows, cols 
    );
    auto row_grid = (rows>>4 <12000 )? rows>>4:rows>>6;
    dim3 grid(1, row_grid);
    dim3 block(16,16);
	RMSNormImpl_bp<256><<<grid,block>>>(
		(__nv_bfloat16*)dinp, 
        ( const __nv_bfloat16*)dout, 
        ( const __nv_bfloat16*)x_rmsnorm,  ( const __nv_bfloat16*)weight,  inv_variance,
        rows, cols
	);
}
