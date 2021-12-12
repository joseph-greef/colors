
#ifndef IMGPROC_H
#define IMGPROC_H

#include "buffer.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <class T>
__host__ __device__ Pixel<T> interpolate(float x, float y, Buffer<Pixel<T>> &buffer);
template <class T>
__host__ __device__ T truncate(T value);

#include "imgproc.h.cu"

#endif //IMGPROC_H

