
#ifndef ROTATION_H
#define ROTATION_H

#include "transformation.h"

class Rotation : public Transformation {
private:
    float center_x_;
    float center_y_;

    float rotation_amount_;
public:
    Rotation(int width, int height);
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //ROTATION_H
