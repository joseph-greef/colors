
#include <cmath>
#include <iostream>
#include <random>

#include "noise.h"

Noise::Noise(int width, int height)
    : Transformation(width, height)
{
    noise_type_ = dist_positive_(e2_) * 2;
    if(noise_type_ == 0) {
        amplitude_ = dist_positive_(e2_) * 70;
        bias_ = dist_full_(e2_) * 2;
        amp_mod_ = 2 * amplitude_ + 1;
    }
    else if(noise_type_ == 1) {
        amplitude_ = dist_positive_(e2_) * 70;
        bias_ = dist_full_(e2_) * 2;
        amp_mod_ = 2 * amplitude_ + 1;
    }
}

void Noise::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(noise_type_ == 0) {
        for(int y = 0; y < height_; y++) {
            for(int x = 0; x < width_; x++) {
                uint32_t src = current_frame[y * width_ + x].value;
                uint32_t *dest = &target_frame[y * width_ + x].value;
                *dest = 0;

                for(int i = 0; i < 4; i++) {
                    int16_t mod = (rand() % (amp_mod_)) - amplitude_ + bias_;
                    int16_t temp = (src & 0xFF) + mod;
                    if(temp < 0) {
                        temp = 0;
                    }
                    else if(temp > 255) {
                        temp = 255;
                    }

                    *dest += (uint32_t)temp << (i * 8);
                    src >>= 8;
                }
            }
        }
    }
    else if(noise_type_ == 1) {
        for(int y = 0; y < height_; y++) {
            for(int x = 0; x < width_; x++) {
                uint32_t src = current_frame[y * width_ + x].value;
                uint32_t *dest = &target_frame[y * width_ + x].value;
                int color_to_rand = rand() % 4;
                int32_t temp = (src >> (color_to_rand * 8)) & 0xFF;

                temp += (rand() % (amp_mod_)) - amplitude_ + bias_;
                if(temp < 0) {
                    temp = 0;
                }
                else if(temp > 255) {
                    temp = 255;
                }


                src &= ~(0xFF << (color_to_rand * 8));
                src += temp << (color_to_rand * 8);

                *dest = src;

            }
        }
    }
}
