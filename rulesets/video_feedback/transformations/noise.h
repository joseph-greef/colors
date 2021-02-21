
#ifndef NOISE_H
#define NOISE_H

#include "curand.h"
#include "transformation.h"

class Noise : public Transformation {
private:
    uint32_t amplitude_;
    uint8_t amp_mod_;
    uint8_t bias_;
    int noise_type_;
    curandGenerator_t curand_gen_;
    uint32_t *cudev_random_numbers_;
    uint32_t *random_numbers_;
    int num_randoms_;
public:
    Noise(int width, int height);
    ~Noise();
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //NOISE_H
