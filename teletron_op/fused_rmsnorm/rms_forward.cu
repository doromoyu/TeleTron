#include <cuda_bf16.h>
#include <ATen/ATen.h>

template<typename T>
__inline__ __device__ T Rsqrt(T x);

template<>
__inline__ __device__ float Rsqrt<float>(float x) {

  return rsqrt(x);

}

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

template<const int block_size = 256>
__global__ void RMSNormImpl(__nv_bfloat16* output, 
                                       const __nv_bfloat16* input, 
									   const __nv_bfloat16* weight, 
									   const int64_t rows,
                                       const int64_t cols, 
									   const double epsilon_d, 
                                       float* inv_variance) {

    
    __shared__ float sum_sm[16];

    constexpr int pack_size = 8;
    const int tidx = threadIdx.x;   
    const int tidy = threadIdx.y;   
    const int num_packs_per_row = cols / pack_size;
    const float epsilon = static_cast<float>(epsilon_d); 

    for(int64_t row = blockIdx.y * blockDim.y + tidy; row < rows; row += gridDim.y * blockDim.y)
    {
        const __nv_bfloat16* current_input_row = input + row * cols;
        __nv_bfloat16* current_output_row = output + row * cols;

        float x2_pack_sum = 0.0f;
       
            const __nv_bfloat16* input_pack_ptr = current_input_row + tidx * pack_size;
            const __nv_bfloat162* input_bf162_ptr = reinterpret_cast<const __nv_bfloat162*>(input_pack_ptr);

           __nv_bfloat162 pack_data1[pack_size / 2];
           *reinterpret_cast<float4*>(pack_data1) = *reinterpret_cast<float4*>((const_cast<__nv_bfloat162*>(input_bf162_ptr)));
           #pragma unroll
           for (int i = 0; i < pack_size / 2; i++){
                
                auto tmp_f2 = __bfloat1622float2(pack_data1[i]);
                x2_pack_sum += tmp_f2.x * tmp_f2.x + tmp_f2.y * tmp_f2.y;
           }

        blockReduceSum(x2_pack_sum, sum_sm);
        float inv_mean = Rsqrt(max(Div<float>(sum_sm[tidy], cols), 0.0f) + epsilon);
        if (tidx == 0) {
			inv_variance[row] = inv_mean;
		}

            const int col_offset = tidx* pack_size;
            __nv_bfloat16* output_pack_ptr = current_output_row + col_offset;
            const __nv_bfloat16* x_weight_pack_ptr = weight + col_offset;

            __nv_bfloat162 pack_data[pack_size>>1];
            *reinterpret_cast<float4*>(pack_data) = *reinterpret_cast<float4*>(const_cast<__nv_bfloat16*>(x_weight_pack_ptr));
            #pragma unroll
            for (int i = 0; i < pack_size>>1; ++i) {
                
                auto val_bf2 =  pack_data1[i];
                auto val_f2=__bfloat1622float2(val_bf2);
                val_f2.x  = val_f2.x * inv_mean;
                val_f2.y  = val_f2.y * inv_mean;
                pack_data[i] = __hmul2(__float22bfloat162_rn(val_f2), pack_data[i]);
            }
            *reinterpret_cast<float4*>(output_pack_ptr) = *reinterpret_cast<float4*>(pack_data);
    }
}

void launch_rms_forward(
	at::BFloat16 *  output,     
    const at::BFloat16 *  input, 
    const at::BFloat16 *  weight, 
    const int64_t rows,         
    const int64_t cols,         
    const double epsilon,       
    float* inv_variance) {

	auto row_grid = (rows>>4 <12000 )? rows>>4:rows>>6;
	dim3 grid(1, row_grid);
    dim3 block(16,16);
	RMSNormImpl<256><<<grid,block>>>(
		(__nv_bfloat16*)output,
		( const __nv_bfloat16*)input,
		( const __nv_bfloat16*)weight,
		rows, cols, epsilon, 
    	inv_variance
	);
} 