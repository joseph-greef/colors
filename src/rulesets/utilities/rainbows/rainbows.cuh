
#ifndef _RAINBOWS_CUH
#define _RAINBOWS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "buffer.cuh"


__host__ __device__
void age_to_pixels_step(Buffer<int> *board, Buffer<Pixel<uint8_t>> *pixels, int index,
                        uint32_t *alive_gradient, uint32_t *dead_gradient,
                        int alive_offset, int dead_offset, int color_offset,
                        bool changing_background);

void call_age_to_pixels_kernel(Buffer<int> *board, Buffer<Pixel<uint8_t>> *pixels,
                               int alive_color_scheme, int dead_color_scheme,
                               int alive_offset, int dead_offset, int color_offset,
                               bool changing_background);

#endif //_RAINBOWS_CUH
