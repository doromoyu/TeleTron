#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template<typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 16; mask > 0; mask >>=1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<int blockSize=1024>
__device__ inline float blockReduceSum(float val) {
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
    return val;
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
  return a / b;

}



__global__ void RMSNorm_weight1_bp_opt(
    float* dweight_tmp,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols
) {
    constexpr int pack_size = 8; 


    const int tid = threadIdx.y;
    const int block_size = blockDim.y * gridDim.y;
    const int col_start = blockIdx.x * pack_size;

    using LoadStoreType = float4;

    float thread_dweight[pack_size] = {0.f};

    for (int64_t row = tid + blockIdx.y * blockDim.y; row < rows; row += block_size) {
        const int row_offset = row * cols;
        auto cur_dout_packed =  *reinterpret_cast<const LoadStoreType*>(dout + row_offset + col_start);
        auto cur_x_norm_packed = *reinterpret_cast<const LoadStoreType*>(x_rmsnorm + row_offset + col_start);
        auto cur_weight_packed = *reinterpret_cast<const LoadStoreType*>(weight + col_start);

        float2 d01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.x));
        float2 xn01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.x));
        float2 w01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.x));
        thread_dweight[0] += Div(d01.x * xn01.x, w01.x);
        thread_dweight[1] += Div(d01.y * xn01.y, w01.y);

        float2 d23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.y));
        float2 xn23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.y));
        float2 w23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.y));
        thread_dweight[2] += Div(d23.x * xn23.x, w23.x);
        thread_dweight[3] += Div(d23.y * xn23.y, w23.y);

        float2 d45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.z));
        float2 xn45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.z));
        float2 w45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.z));
        thread_dweight[4] += Div(d45.x * xn45.x, w45.x);
        thread_dweight[5] += Div(d45.y * xn45.y, w45.y);

        float2 d67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.w));
        float2 xn67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.w));
        float2 w67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.w));
        thread_dweight[6] += Div(d67.x * xn67.x, w67.x);
        thread_dweight[7] += Div(d67.y * xn67.y, w67.y);
    }

    #pragma unroll
    for (int i = 0; i < pack_size; ++i) {
        thread_dweight[i] = blockReduceSum(thread_dweight[i]);
    }
    if (tid == 0) {
        *reinterpret_cast<LoadStoreType*>(dweight_tmp + blockIdx.y * cols + col_start) = *reinterpret_cast<LoadStoreType*>(thread_dweight);
        *reinterpret_cast<LoadStoreType*>(dweight_tmp + blockIdx.y * cols + col_start + 4) = *reinterpret_cast<LoadStoreType*>(&(thread_dweight[4]));
    }
}


__global__ void RMSNorm_weight2_bp_opt(
    __nv_bfloat16* dweight,
    float* dweight_tmp, 
    const int64_t rows, const int64_t cols
) {
    constexpr int pack_size = 4; 

    const int xid = threadIdx.x / 32;
    const int yid = threadIdx.x & 31;

    const int row_offset = yid * cols;
    const int col_offset = xid * 4;

    float4 dweight_f = *reinterpret_cast<float4*>(dweight_tmp + row_offset + col_offset);

    dweight_f.x = warpReduceSum<float>(dweight_f.x);
    dweight_f.y = warpReduceSum<float>(dweight_f.y);
    dweight_f.z = warpReduceSum<float>(dweight_f.z);
    dweight_f.w = warpReduceSum<float>(dweight_f.w);

    if (yid == 0)
    {
        __nv_bfloat162 tmp[2];
        tmp[0] = __floats2bfloat162_rn(dweight_f.x, dweight_f.y);
        tmp[1] = __floats2bfloat162_rn(dweight_f.z, dweight_f.w);

        *reinterpret_cast<float2*>(dweight + col_offset) = *reinterpret_cast<float2*>(tmp);
    }
}

__global__ void RMSNorm_weight_micro_bp_opt(
    __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols
) {
    constexpr int pack_size = 8;
    const int tid = threadIdx.y;
    const int block_size = blockDim.y; 
    const int col_start = blockIdx.x * pack_size;

    using LoadStoreType = float4;

    float thread_dweight[pack_size] = {0.f};

    for (int64_t row = tid; row < rows; row += block_size) {
        const int row_offset = row * cols;
        auto cur_dout_packed =  *reinterpret_cast<const LoadStoreType*>(dout + row_offset + col_start);
        auto cur_x_norm_packed = *reinterpret_cast<const LoadStoreType*>(x_rmsnorm + row_offset + col_start);
        auto cur_weight_packed = *reinterpret_cast<const LoadStoreType*>(weight + col_start); 


        float2 d01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.x));
        float2 xn01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.x));
        float2 w01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.x));
        thread_dweight[0] += Div(d01.x * xn01.x, w01.x);
        thread_dweight[1] += Div(d01.y * xn01.y, w01.y);


        float2 d23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.y));
        float2 xn23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.y));
        float2 w23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.y));
        thread_dweight[2] += Div(d23.x * xn23.x, w23.x);
        thread_dweight[3] += Div(d23.y * xn23.y, w23.y);

        float2 d45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.z));
        float2 xn45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.z));
        float2 w45 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.z));
        thread_dweight[4] += Div(d45.x * xn45.x, w45.x);
        thread_dweight[5] += Div(d45.y * xn45.y, w45.y);

        float2 d67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_dout_packed.w));
        float2 xn67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_x_norm_packed.w));
        float2 w67 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&cur_weight_packed.w));
        thread_dweight[6] += Div(d67.x * xn67.x, w67.x);
        thread_dweight[7] += Div(d67.y * xn67.y, w67.y);
    }
    #pragma unroll
    for (int i = 0; i < pack_size; ++i) {
        thread_dweight[i] = blockReduceSum(thread_dweight[i]);
    }

    if ( tid == 0 )
    {
        __nv_bfloat162 final_dweight_packed[4];

        final_dweight_packed[0] = (__floats2bfloat162_rn(thread_dweight[0], thread_dweight[1]));
        final_dweight_packed[1] = (__floats2bfloat162_rn(thread_dweight[2], thread_dweight[3]));
        final_dweight_packed[2] = (__floats2bfloat162_rn(thread_dweight[4], thread_dweight[5]));
        final_dweight_packed[3] = (__floats2bfloat162_rn(thread_dweight[6], thread_dweight[7]));

        *reinterpret_cast<LoadStoreType*>(dweight + col_start) = *reinterpret_cast<LoadStoreType*>(final_dweight_packed);
    }
}


void launch_rms_weight_backward(
    __nv_bfloat16* dweight,
    const __nv_bfloat16* dout, const __nv_bfloat16* x_rmsnorm, const __nv_bfloat16* weight,
    const int64_t rows, const int64_t cols)
{
    if (rows < 16384) 
    {
        dim3 block1(1,1024);
        dim3 grid1(16,1);
        RMSNorm_weight_micro_bp_opt<<<grid1, block1>>>(
            dweight, dout, x_rmsnorm, weight, rows, cols
        );
    } else {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    

    const int tmp_rows = 32;  
    auto dweight_tmp = at::empty({tmp_rows, cols}, 
        at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    constexpr int pack_size1 = 8;
    const int grid_dim_x1 = cols / pack_size1;  
    const int grid_dim_y1 = 32;               
    
    dim3 grid1(grid_dim_x1, grid_dim_y1);
    dim3 block1(1, 1024);  
    
    RMSNorm_weight1_bp_opt<<<grid1, block1, 0, stream>>>(
        dweight_tmp.data_ptr<float>(),
        dout,
        x_rmsnorm,
        weight,
        rows,
        cols
    );

    constexpr int pack_size2 = 4;
    dim3 grid2(1);
    dim3 block2(1024); 
    

    RMSNorm_weight2_bp_opt<<<grid2, block2, 0, stream>>>(
        dweight,
        dweight_tmp.data_ptr<float>(),
        tmp_rows, 
        cols    
    );
    }
}
