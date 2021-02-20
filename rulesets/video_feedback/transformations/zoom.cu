
#include <cmath>

#include "zoom.h"

#include "cuda_runtime.h"
#include "imgproc.cuh"

__global__ static void transformation(float scale_x, float scale_y,
                                      float center_x, float center_y,
                                      int width, int height,
                                      Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame) {
}

Zoom::Zoom(int width, int height)
    : Transformation(width, height)
{
    scale_x_ = 1 - dist_positive_(e2_) / 80;
    scale_y_ = 1 - dist_positive_(e2_) / 80;
    center_x_ = width / 2 + dist_full_(e2_) * width_ / 10;
    center_y_ = height / 2 + dist_full_(e2_) * height_ / 10;

}

void Zoom::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(use_gpu) {
        transformation<<<512, 128>>>(scale_x_, scale_y_, center_x_, center_y_,
                                     width_, height_,
                                     last_frame, current_frame, target_frame);
    }
    else {
        for(int target_y = 0; target_y < height_; target_y++) {
            for(int target_x = 0; target_x < width_; target_x++) {
                float current_x = (target_x - center_x_) * scale_x_ + center_x_;
                float current_y = (target_y - center_y_) * scale_y_ + center_y_;

                target_frame[target_y * width_ + target_x] =
                        interpolate(current_x, current_y,
                                    width_, height_, current_frame);
            }
        }
    }
}
