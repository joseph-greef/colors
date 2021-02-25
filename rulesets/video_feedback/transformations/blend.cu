
#include <cmath>

#include "blend.h"

#include "cuda_runtime.h"


__host__ __device__ static void transformation(Pixel *last_frame, Pixel *current_frame, 
                                               Pixel *target_frame, int index) {
    uint32_t last = last_frame[index].value;
    uint32_t current = current_frame[index].value;
    uint32_t *target = &target_frame[index].value;

    *target = 0;
    for(int i = 0; i < 4; i++) {
        uint8_t temp = ((last & 0xFF)/2) + ((current & 0xFF)/2);

        *target += (uint32_t)temp << (i * 8);

        last >>= 8;
        current >>= 8;
    }


}
__global__ static
    void transformation_kernel(int width, int height,
                               Pixel *last_frame, Pixel *current_frame,
                               Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        transformation(last_frame, current_frame, target_frame, index);

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
        transformation_kernel<<<512, 128>>>(width_, height_, last_frame,
                                            current_frame, target_frame);
    }
    else {
        for(int index = 0; index < height_ * width_; index++) {
            transformation(last_frame, current_frame, target_frame, index);
        }
    }
}

std::string Blend::get_rule_string() {
    std::ostringstream oss;
    oss << "blend:";
    return oss.str();
}

