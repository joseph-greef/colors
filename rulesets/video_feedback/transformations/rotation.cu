
#include <algorithm>
#include <cmath>

#include "rotation.h"

#include "cuda_runtime.h"

__global__ static void transformation(float rotation_amount,
                                      float center_x, float center_y,
                                      int width, int height,
                                      Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        int target_x = index % width;
        int target_y = index / width;

        float new_x = (target_x - center_x) * cos(rotation_amount) -
                      (target_y - center_y) * sin(rotation_amount) +
                      center_x;

        float new_y = (target_x - center_x) * sin(rotation_amount) +
                      (target_y - center_y) * cos(rotation_amount) +
                      center_y;

        new_x = new_x < 0 ? 0 : new_x;
        new_x = new_x >= width? width-1 : new_x;
        new_y = new_y < 0 ? 0 : new_y;
        new_y = new_y >= height? height-1 : new_y;

        target_frame[target_y * width + target_x] =
                interpolate(new_x, new_y, width, height, current_frame);

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
                float new_x = (target_x - center_x_) * std::cos(rotation_amount_) -
                              (target_y - center_y_) * std::sin(rotation_amount_) +
                              center_x_;

                float new_y = (target_x - center_x_) * std::sin(rotation_amount_) +
                              (target_y - center_y_) * std::cos(rotation_amount_) +
                              center_y_;

                new_x = new_x < 0 ? 0 : new_x;
                new_x = new_x >= width_ ? width_-1 : new_x;
                new_y = new_y < 0 ? 0 : new_y;
                new_y = new_y >= height_ ? height_-1 : new_y;

                target_frame[target_y * width_ + target_x] =
                        interpolate(new_x, new_y, width_, height_, current_frame);

            }
        }
    }
}
