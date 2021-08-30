// challenge: implement for multiple input types.

// simple element-wise operation
template<typename T>
__global__ void sign_op_gpu(int n, T* in, T*out){
    for(int i=0; i<n; i++){
        out[i] = (in[i] > 0) ? 1 : ( (in[i] < 0) ? -1 : 0 );
    }
}

// TODO implementation of better parallelism
template<typename T>
void signOpLauncher(int n, T* in, T* out){
    sign_op_gpu<<<1,1>>>(n, in, out);
}