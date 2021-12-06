
#include "rainbows.h"
#include "rainbows.cuh"

__device__ static uint32_t colors_device[][RAINBOW_LENGTH] = {
#include "gradients.h"
};

__host__ __device__
void age_to_pixels_step(Board<int> *board, uint32_t *pixels, int index,
                        uint32_t *alive_gradient, uint32_t *dead_gradient,
                        int alive_offset, int dead_offset, int color_offset,
                        bool changing_background) {
    if(board->get(index) > 0) {
        pixels[index] = alive_gradient
                [(board->get(index) + alive_offset + color_offset) &
                 255];
    }
    else if(changing_background || board->get(index) < 0) {
        pixels[index] = dead_gradient
                [(-board->get(index) + dead_offset + color_offset) &
                 255];
    }
    else {
        pixels[index] = dead_gradient
                [(-board->get(index) + dead_offset) & 255];
    }
}

__global__ static
void age_to_pixels_kernel(Board<int> *board, uint32_t *pixels,
                          int alive_color_scheme, int dead_color_scheme,
                          int alive_offset, int dead_offset, int color_offset,
                          bool changing_background) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < board->height_ * board->width_) {
        age_to_pixels_step(board, pixels, index,
                           colors_device[alive_color_scheme],
                           colors_device[dead_color_scheme], alive_offset,
                           dead_offset, color_offset, changing_background);

        index += blockDim.x * gridDim.x;
    }
}

void call_age_to_pixels_kernel(Board<int> *board, uint32_t *pixels,
                               int alive_color_scheme, int dead_color_scheme,
                               int alive_offset, int dead_offset, int color_offset,
                               bool changing_background) {
        age_to_pixels_kernel<<<512, 128>>>(board, pixels,
                                       alive_color_scheme, dead_color_scheme,
                                       alive_offset, dead_offset, color_offset,
                                       changing_background);
}

