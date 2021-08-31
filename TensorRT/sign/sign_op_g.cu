// challenge: implement for multiple input types.

// simple element-wise operation
template<typename T>
__global__ void sign_op_gpu(int n, const T* in, T* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;                
    // CUDA code for element-wise operation
    for(int i=index; i<n; i+=stride){
        out[i] = (in[i] > 0) ? 1 : ( (in[i] < 0) ? -1 : 0 );
    }
}

// Unable to template this?
template<typename T>
void signOpLauncher(int n, const T* in, T* out){
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    sign_op_gpu<<<numBlocks, blockSize>>>(n, in, out);
}

// void signOpLauncher(int n, const float* in, float* out){
//     sign_op_gpu<<<1,1>>>(n, in, out);
// }

// void signOpLauncher(int n, const int32_t* in, int32_t* out){
//     sign_op_gpu<<<1,1>>>(n, in, out);
// }