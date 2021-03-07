
#include <algorithm>
#include <cmath>
#include <iostream>

#include "rotation.h"

#include "cuda_runtime.h"

__host__ __device__ static
    void transformation(float rotation_amount, float center_x, float center_y,
                        int width, int height,
                        Pixel *last_frame, Pixel *current_frame, Pixel *target_frame,
                        int target_x, int target_y) {
    float new_x = (target_x - center_x) * cos(rotation_amount) -
                  (target_y - center_y) * sin(rotation_amount) +
                  center_x;

    float new_y = (target_x - center_x) * sin(rotation_amount) +
                  (target_y - center_y) * cos(rotation_amount) +
                  center_y;

    if(new_x < 0 || new_x >= width || new_y < 0 || new_y >= height) {
        target_frame[target_y * width + target_x].value = 0;
    }
    else {
        target_frame[target_y * width + target_x] =
                interpolate(new_x, new_y, width, height, current_frame);
    }

}

__global__ static void transformation(float rotation_amount,
                                      float center_x, float center_y,
                                      int width, int height,
                                      Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        int target_x = index % width;
        int target_y = index / width;

        transformation(rotation_amount, center_x, center_y, width, height,
                       last_frame, current_frame, target_frame,
                       target_x, target_y);

        index += blockDim.x * gridDim.x;
    }
}

Rotation::Rotation(int width, int height)
    : Transformation(width, height)
{
    center_x_ = width / 2 + dist_full_(e2_) * 50;
    center_y_ = height / 2 + dist_full_(e2_) * 50;

    rotation_amount_ = dist_full_(e2_) * 3.14159265358979323846 / 100;
    rotation_amount_ += static_cast<int>(dist_positive_(e2_) * 2) * 3.141592;
}

Rotation::Rotation(int width, int height, std::string params)
    : Transformation(width, height)
{
        size_t next_delim;
        std::string params_copy(params);
        std::string param;

        next_delim = params_copy.find(',');
        param = params_copy.substr(0, next_delim);
        center_x_ = stof(param);
        params_copy.erase(0, next_delim + 1);

        next_delim = params_copy.find(',');
        param = params_copy.substr(0, next_delim);
        center_y_ = stof(param);
        params_copy.erase(0, next_delim + 1);

        next_delim = params_copy.find(',');
        param = params_copy.substr(0, next_delim);
        rotation_amount_ = stof(param);
        params_copy.erase(0, next_delim + 1);
}

void Rotation::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(use_gpu) {
        transformation<<<512, 128>>>(rotation_amount_, center_x_, center_y_,
                                     width_, height_,
                                     last_frame, current_frame, target_frame);
    }
    else {
        for(int target_y = 0; target_y < height_; target_y++) {
            for(int target_x = 0; target_x < width_; target_x++) {
                transformation(rotation_amount_, center_x_, center_y_, width_, height_,
                               last_frame, current_frame, target_frame,
                               target_x, target_y);
            }
        }
    }
}

std::string Rotation::get_rule_string() {
    std::ostringstream oss;
    oss << "rotation:" << center_x_ << "," << center_y_ << "," <<rotation_amount_;
    return oss.str();
}

