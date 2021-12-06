
#include "empty_ruleset.cuh"


__global__ void cuda_empty_ruleset() {
    //unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    //while (index < height * width) {

        //int x = index % width;
        //int y = index / width;

        //index += blockDim.x * gridDim.x;
    //}
}

void call_cuda_empty_ruleset() {
    cuda_empty_ruleset<<<512, 128>>>();
}
