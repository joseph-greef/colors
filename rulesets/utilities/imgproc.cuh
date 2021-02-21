
#ifndef IMGPROC_H
#define IMGPROC_H

#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

union Pixel {
    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    } part;
    uint32_t value;

};

static_assert(sizeof(Pixel) == sizeof(uint32_t));

__host__ __device__ Pixel interpolate(float x, float y, int width, int height, Pixel *buffer);
__host__ __device__ uint8_t truncate(int32_t value);

#endif //IMGPROC_H

