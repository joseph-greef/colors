
#ifndef BLEND_H
#define BLEND_H

#include "transformation.h"

class Blend : public Transformation {
private:
public:
    Blend(int width, int height);
    Blend(int width, int height, std::string params);
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
    std::string get_rule_string();
};

#endif //BLEND_H
