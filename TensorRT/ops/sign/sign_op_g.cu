// simple element-wise operation for Sign.

// Future work: Use templates to generalise the code for other data types.

// for int32_t:
__global__ void sign_op_gpu(int n, const int32_t* in, int32_t* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // CUDA code for element-wise operation
    for(int i=index; i<n; i+=stride){
        out[i] = (in[i] > 0) ? 1 : ( (in[i] < 0) ? -1 : 0 );
    }
}
void signOpLauncher(int n, const int32_t* in, int32_t* out){
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sign_op_gpu<<<numBlocks, blockSize>>>(n, in, out);
}

// for float:
__global__ void sign_op_gpu(int n, const float* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // CUDA code for element-wise operation
    for(int i=index; i<n; i+=stride){
        out[i] = (in[i] > 0) ? 1 : ( (in[i] < 0) ? -1 : 0 );
    }
}
void signOpLauncher(int n, const float* in, float* out){
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sign_op_gpu<<<numBlocks, blockSize>>>(n, in, out);
}