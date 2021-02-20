
#ifndef NOISE_H
#define NOISE_H

#include "transformation.h"

class Noise : public Transformation {
private:
    uint8_t amplitude_;
    uint8_t amp_mod_;
    uint8_t bias_;
    int noise_type_;
public:
    Noise(int width, int height);
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //NOISE_H
