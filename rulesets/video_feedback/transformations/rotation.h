
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
    Rotation(int width, int height, std::string params);
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
    std::string get_rule_string();
};

#endif //ROTATION_H
