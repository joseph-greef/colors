
#include <cmath>

#include "blend.h"

Blend::Blend(int width, int height)
    : Transformation(width, height)
{
}

void Blend::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    for(int y = 0; y < height_; y++) {
        for(int x = 0; x < width_; x++) {
            uint32_t last = last_frame[y * width_ + x].value;
            uint32_t current = current_frame[y * width_ + x].value;
            uint32_t *target = &target_frame[y * width_ + x].value;

            *target = 0;
            for(int i = 0; i < 4; i++) {
                uint8_t temp = ((last & 0xFF)/2) + ((current & 0xFF)/2);

                *target += (uint32_t)temp << (i * 8);

                last >>= 8;
                current >>= 8;
            }
        }
    }
}
