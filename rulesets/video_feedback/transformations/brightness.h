
#ifndef BRIGHTNESS_H
#define BRIGHTNESS_H

#include "curand.h"
#include "transformation.h"

class Brightness : public Transformation {
private:
    int32_t amplitude_;
public:
    Brightness(int width, int height);
    ~Brightness();
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //BRIGHTNESS_H
