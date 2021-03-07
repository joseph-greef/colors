
#include <cmath>
#include <iostream>

#include "zoom.h"

#include "cuda_runtime.h"
#include "imgproc.cuh"

__host__ __device__ static
    void transformation(float scale_x, float scale_y, float center_x, float center_y,
                        Board<Pixel<float>> &last_frame, Board<Pixel<float>> &current_frame, Board<Pixel<float>> &target_frame,
                        int target_x, int target_y) {
    float current_x = (target_x - center_x) * scale_x + center_x;
    float current_y = (target_y - center_y) * scale_y + center_y;

    target_frame.set(target_x, target_y,
            interpolate<float>(current_x, current_y, current_frame));
}

__global__ static void transformation_kernel(float scale_x, float scale_y,
                                             float center_x, float center_y,
                                             Board<Pixel<float>> last_frame, Board<Pixel<float>> current_frame,
                                             Board<Pixel<float>> target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < target_frame.height_ * target_frame.width_) {
        int target_x = index % target_frame.width_;
        int target_y = index / target_frame.width_;

        transformation(scale_y, scale_y, center_y, center_y,
                       last_frame, current_frame, target_frame,
                       target_x, target_y);

        index += blockDim.x * gridDim.x;
    }
}

Zoom::Zoom(int width, int height)
    : Transformation(width, height)
{
    scale_x_ = 1 - dist_positive_(e2_) / 180;
    scale_y_ = 1 - dist_positive_(e2_) / 180;
    center_x_ = width / 2 + dist_full_(e2_) * width_ / 10;
    center_y_ = height / 2 + dist_full_(e2_) * height_ / 10;

}

Zoom::Zoom(int width, int height, std::string params)
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
        scale_x_ = stof(param);
        params_copy.erase(0, next_delim + 1);

        next_delim = params_copy.find(',');
        param = params_copy.substr(0, next_delim);
        scale_y_ = stof(param);
        params_copy.erase(0, next_delim + 1);
}

void Zoom::apply_transformation(Board<Pixel<float>> &last_frame,
                                Board<Pixel<float>> &current_frame,
                                Board<Pixel<float>> &target_frame, bool use_gpu) {
    if(use_gpu) {
        transformation_kernel<<<512, 128>>>(scale_x_, scale_y_, center_x_, center_y_,
                                     last_frame, current_frame, target_frame);
    }
    else {
        for(int target_y = 0; target_y < height_; target_y++) {
            for(int target_x = 0; target_x < width_; target_x++) {
                transformation(scale_y_, scale_y_, center_y_, center_y_,
                               last_frame, current_frame, target_frame,
                               target_x, target_y);
            }
        }
    }
}

std::string Zoom::get_rule_string() {
    std::ostringstream oss;
    oss << "zoom:" << center_x_ << "," << center_y_ << "," <<
                      scale_x_ << "," << scale_y_;
    return oss.str();
}

