
#include "painting.cuh"


__global__ void cuda_painting() {
    /*
    //unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < buffer->h_ * buffer->w_) {
        //int x = index % buffer->w_;
        //int y = index / buffer->w_;

        index += blockDim.x * gridDim.x;
    }
    */
}

void call_cuda_painting() {
    cuda_painting<<<512, 128>>>();
}
