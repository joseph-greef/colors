
#include <algorithm>
#include <cmath>

#include "rotation.h"

Rotation::Rotation(int width, int height)
    : Transformation(width, height)
{
    center_x_ = width / 2 + dist_full_(e2_) * 50;
    center_y_ = height / 2 + dist_full_(e2_) * 50;

    rotation_amount_ = dist_full_(e2_) * 3.14159265358979323846 / 48;
}

void Rotation::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(use_gpu) {
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
