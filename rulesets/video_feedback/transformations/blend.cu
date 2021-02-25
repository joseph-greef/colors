
#include <cmath>

#include "blend.h"

#include "cuda_runtime.h"

__global__ static void transformation(int width, int height,
                                      Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        int x = index % width;
        int y = index / width;

        uint32_t last = last_frame[y * width + x].value;
        uint32_t current = current_frame[y * width + x].value;
        uint32_t *target = &target_frame[y * width + x].value;

        *target = 0;
        for(int i = 0; i < 4; i++) {
            uint8_t temp = ((last & 0xFF)/2) + ((current & 0xFF)/2);

            *target += (uint32_t)temp << (i * 8);

            last >>= 8;
            current >>= 8;
        }

        index += blockDim.x * gridDim.x;
    }
}

Blend::Blend(int width, int height)
    : Transformation(width, height)
{
}

Blend::Blend(int width, int height, std::string params)
    : Transformation(width, height)
{
}

void Blend::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    if(use_gpu) {
        transformation<<<512, 128>>>(width_, height_,
                                     last_frame, current_frame, target_frame);
    }
    else {
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
}

std::string Blend::get_rule_string() {
    std::ostringstream oss;
    oss << "blend:";
    return oss.str();
}

