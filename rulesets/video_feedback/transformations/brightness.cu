
#include <cmath>
#include <iostream>
#include <random>

#include "brightness.h"
#include "imgproc.cuh"

#include "cuda_runtime.h"


__host__ __device__ static
    void transformation(int32_t amplitude,
                        Pixel *current_frame, Pixel *target_frame, int index)
{
    Pixel target = { 0 };
    Pixel current = current_frame[index];

    target.part.r = truncate(((int32_t)current.part.r) + amplitude);
    target.part.g = truncate(((int32_t)current.part.g) + amplitude);
    target.part.b = truncate(((int32_t)current.part.b) + amplitude);

    target_frame[index] = target;
}

__global__ static
    void transformation_kernel(int amplitude,
                               int width, int height,
                               Pixel *last_frame, Pixel *current_frame,
                               Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        transformation(amplitude,
                       current_frame, target_frame, index);

        index += blockDim.x * gridDim.x;
    }
}


Brightness::Brightness(int width, int height)
    : Transformation(width, height)
{
    amplitude_ = 1;//static_cast<int>(dist_full_(e2_) * 5) +
                 //static_cast<int>(dist_positive_(e2_) * 5);
}

Brightness::Brightness(int width, int height, std::string params)
    : Transformation(width, height)
{
        size_t next_delim;
        std::string params_copy(params);
        std::string param;

        next_delim = params_copy.find(',');
        param = params_copy.substr(0, next_delim);
        amplitude_ = stoi(param);
        params_copy.erase(0, next_delim + 1);
}

Brightness::~Brightness() {
}

void Brightness::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(use_gpu) {
        transformation_kernel<<<512, 128>>>(amplitude_,
                                            width_, height_,
                                            last_frame, current_frame, target_frame);
    }
    else {
        for(int index = 0; index < height_ * width_; index++) {
            transformation(amplitude_,
                           current_frame, target_frame, index);
        }
    }
}

std::string Brightness::get_rule_string() {
    std::ostringstream oss;
    oss << "brightness:" << amplitude_;
    return oss.str();
}

