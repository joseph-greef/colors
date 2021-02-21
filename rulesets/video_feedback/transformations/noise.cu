
#include <cmath>
#include <iostream>
#include <random>

#include "noise.h"

#include "cuda_runtime.h"


__host__ __device__ static
    void transformation(uint32_t *random_numbers, int noise_type,
                        uint32_t amplitude_,
                        Pixel *current_frame, Pixel *target_frame, int index)
{
    if(noise_type == 0) {
        if(random_numbers[index*2] % amplitude_ == 0) {
            target_frame[index].value = random_numbers[index*2+1];
        }
        else {
            target_frame[index] = current_frame[index];
        }
    }
    else if(noise_type == 1) {
        target_frame[index] = current_frame[index];
    }
    /*
    if(noise_type == 0) {
        uint32_t src = current_frame[y * width + x].value;
        uint32_t *dest = &target_frame[y * width + x].value;
        *dest = 0;

        for(int i = 0; i < 4; i++) {
            int16_t mod = (rand() % (amp_mod)) - amplitude + bias;
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
    else if(noise_type == 1) {
        uint32_t src = current_frame[y * width + x].value;
        uint32_t *dest = &target_frame[y * width + x].value;
        int color_to_rand = rand() % 4;
        int32_t temp = (src >> (color_to_rand * 8)) & 0xFF;

        temp += (rand() % (amp_mod)) - amplitude + bias;
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
    */
}

__global__ static
    void transformation_kernel(int noise_type, uint32_t *random_numbers,
                               int amplitude,
                               int width, int height,
                               Pixel *last_frame, Pixel *current_frame,
                               Pixel *target_frame) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        transformation(random_numbers, noise_type, amplitude,
                       current_frame, target_frame, index);

        index += blockDim.x * gridDim.x;
    }
}


Noise::Noise(int width, int height)
    : Transformation(width, height)
{
    noise_type_ = dist_positive_(e2_) * 1;
    curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen_, time(NULL));

    if(noise_type_ == 0) {
        num_randoms_ = width_ * height_ * 2;
        amplitude_ = static_cast<int>(dist_positive_(e2_) * 10000);
        //bias_ = dist_full_(e2_) * 2;
        //amp_mod_ = 2 * amplitude_ + 1;
    }
    else if(noise_type_ == 1) {
        num_randoms_ = 1;
        //amplitude_ = dist_positive_(e2_) * 70;
        //bias_ = dist_full_(e2_) * 2;
        //amp_mod_ = 2 * amplitude_ + 1;
    }
    cudaMalloc((void**)&cudev_random_numbers_, num_randoms_ * sizeof(uint32_t));
    random_numbers_ = new uint32_t[num_randoms_];
}

Noise::~Noise() {
    cudaFree(cudev_random_numbers_);
    delete [] random_numbers_;
}

void Noise::apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                Pixel *target_frame, bool use_gpu) {
    curandGenerate(curand_gen_, cudev_random_numbers_, num_randoms_);
    if(use_gpu) {
        transformation_kernel<<<512, 128>>>(noise_type_, cudev_random_numbers_, 
                                            amplitude_,
                                            width_, height_,
                                            last_frame, current_frame, target_frame);
    }
    else {
        cudaMemcpy(random_numbers_, cudev_random_numbers_, num_randoms_ * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        for(int index = 0; index < height_ * width_; index++) {
            transformation(random_numbers_, noise_type_, amplitude_,
                           current_frame, target_frame, index);
        }
    }
}
