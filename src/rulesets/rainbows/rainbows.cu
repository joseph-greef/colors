
#include "rainbows.h"
#include "rainbows.cuh"

__device__ static uint32_t colors_device[][RAINBOW_LENGTH] = {
#include "gradients.h"
};

__global__ static
    void rainbows_kernel(Board<int>* board, Board<int>* board_buffer) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < board->height_ * board->width_) {

        index += blockDim.x * gridDim.x;
    }
}


void call_rainbows_kernel(Board<int> *board, Board<int> *board_buffer) {
    rainbows_kernel<<<512, 128>>>(board->device_copy_, board_buffer->device_copy_);
}

