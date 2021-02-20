
#ifndef BLEND_H
#define BLEND_H

#include "transformation.h"

class Blend : public Transformation {
private:
public:
    Blend(int width, int height);
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //BLEND_H
