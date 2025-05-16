// Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
#include <cuda_bf16.h>
#include <ATen/ATen.h>
constexpr int kWarpSize = 32;

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;
}

template<typename T>
inline __device__ void WelfordCombine(T val, T* mean, T* m2, T* count) {
  *count += 1;
  T delta1 = val - *mean;
  *mean += Div(delta1, *count);
  T delta2 = val - *mean;
  *m2 += delta1 * delta2;
}

template<typename T>
inline __device__ void WelfordCombine(T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == 0) { return; }
  T new_count = *count + b_count;
  T nb_over_n = Div(b_count, new_count);
  T delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                             T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width >>1; mask > 0; mask >>=1) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T* result_mean, T* result_m2, T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  T warp_count = 0;
  WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    T block_count = 0;
    WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}

template<typename T>
__inline__ __device__ T Rsqrt(T x);

template<>
__inline__ __device__ float Rsqrt<float>(float x) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __frsqrt_rn(x);
#else
  return rsqrt(x);
#endif
}



template<const int block_size = 512>
__global__ void AdaLNImpl(__nv_bfloat16* output, 
                                        __nv_bfloat16* x_norm, 
                                       const __nv_bfloat16* input, 
									   const __nv_bfloat16* scale, 
									   const __nv_bfloat16* shift,
									   const int64_t rows,
                                       const int64_t cols, 
									   const double epsilon_d, 
                                       float* inv_variance) {

    __shared__ __nv_bfloat16 buf[3072];

    constexpr int pack_size = 8;
    const int tid = threadIdx.x;
    const float epsilon = static_cast<float>(epsilon_d); 
    const int num_packs_per_row = cols / pack_size;

	for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
		const __nv_bfloat16* current_input_row = input + row * cols;
        __nv_bfloat16* current_output_row = output + row * cols;
        __nv_bfloat16* current_x_norm_row = x_norm + row * cols;

		float thread_mean = 0.0f;
        float thread_m2 = 0.0f;
        float thread_count = 0.0f;

		for (int pack_idx = tid; pack_idx < num_packs_per_row; pack_idx += block_size) {

            const __nv_bfloat16* input_pack_ptr = current_input_row + pack_idx * pack_size;
            const __nv_bfloat162* input_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(input_pack_ptr);


			__nv_bfloat162 pack_data[pack_size / 2]; 
             *reinterpret_cast<float4*>(pack_data) = *reinterpret_cast<float4*>((const_cast<__nv_bfloat162*>(input_bf162_ptr)));

            #pragma unroll
            for(int i = 0; i < pack_size / 2; ++i) {

                 reinterpret_cast<__nv_bfloat162*>(buf)[i * num_packs_per_row + pack_idx] = pack_data[i];

                 WelfordCombine<float>(__bfloat162float(pack_data[i].x), &thread_mean, &thread_m2, &thread_count);
                 WelfordCombine<float>(__bfloat162float(pack_data[i].y), &thread_mean, &thread_m2, &thread_count);
            }
		}
		float row_mean = 0.0f;
		float row_m2 = 0.0f;
		float row_count = 0.0f;
		WelfordBlockAllReduce<float>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2, &row_count);

		float row_variance = max(Div(row_m2, row_count), 0.0f);
		float row_inv_var = Rsqrt(row_variance + epsilon);
		if (tid == 0) {
			inv_variance[row] = row_inv_var;
		}
        
		for (int pack_idx = tid; pack_idx < num_packs_per_row; pack_idx += block_size) {
            const int col_offset = pack_idx * pack_size;
            __nv_bfloat16* output_pack_ptr = current_output_row + col_offset;
            __nv_bfloat16* x_norm_pack_ptr = current_x_norm_row + col_offset;
            const __nv_bfloat162* scale_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(scale + col_offset);
            const __nv_bfloat162* shift_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(shift + col_offset);
			
            __nv_bfloat162 pack_data[2][pack_size>>1];


            #pragma unroll
            for (int i = 0; i < pack_size>>1; ++i) {
                auto val_bf = reinterpret_cast<__nv_bfloat162*>(buf)[i * num_packs_per_row + pack_idx];

                auto valuef =__bfloat1622float2(val_bf);
                valuef.x = (valuef.x - row_mean) * row_inv_var;
                valuef.y = (valuef.y - row_mean) * row_inv_var;

				__nv_bfloat162 scale_bf2 = scale_bf162_ptr[i];
				__nv_bfloat162 shift_bf2 = shift_bf162_ptr[i];

                pack_data[0][i] = __float22bfloat162_rn(valuef);
				pack_data[1][i] = __hfma2(scale_bf2, pack_data[0][i], __hadd2(pack_data[0][i], shift_bf2));
            }
			*reinterpret_cast<float4*>(x_norm_pack_ptr) = *reinterpret_cast<float4*>(pack_data[0]);
			*reinterpret_cast<float4*>(output_pack_ptr) = *reinterpret_cast<float4*>(pack_data[1]);
			
        }
	}
}

void launch_adaln_forward(
	at::BFloat16 *  output,     
    at::BFloat16 * x_norm,
    const at::BFloat16 *  input, 
    const at::BFloat16 *  scale, 
    const at::BFloat16 *  shift, 
    const int64_t rows,         
    const int64_t cols,         
    const double epsilon,       
    float* inv_variance) {

	auto row_grid = (rows <12000 )? rows:rows>>2;
	dim3 grid(row_grid);
    dim3 block(384);
	AdaLNImpl<384><<<grid,block>>>(
		(__nv_bfloat16*)output,
		(__nv_bfloat16*)x_norm, 
		( const __nv_bfloat16*)input,
		( const __nv_bfloat16*)scale,
		( const __nv_bfloat16*)shift,
		rows, cols, epsilon, 
    	inv_variance
	);
} 