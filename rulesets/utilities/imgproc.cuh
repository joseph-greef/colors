
#ifndef IMGPROC_H
#define IMGPROC_H

#include <cstdint>

#include "board.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
union Pixel {
    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    } part;
    uint32_t value;

};
*/

template <class T>
union Pixel {
    struct {
        T r;
        T g;
        T b;
        T a;
    } part;
    T value[4];
};

static_assert(sizeof(Pixel<uint8_t>) == sizeof(uint32_t));

template <class T>
__host__ __device__ Pixel<T> interpolate(float x, float y, Board<Pixel<T>> &board);
template <class T>
__host__ __device__ T truncate(T value);

#include "imgproc.h.cu"

#endif //IMGPROC_H

