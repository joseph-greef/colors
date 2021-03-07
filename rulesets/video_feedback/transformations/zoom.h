
#ifndef ZOOM_H
#define ZOOM_H

#include "transformation.h"

class Zoom : public Transformation {
private:
    float scale_x_;
    float scale_y_;
    float center_x_;
    float center_y_;
public:
    Zoom(int width, int height);
    Zoom(int width, int height, std::string params);
    void apply_transformation(Board<Pixel<float>> &last_frame,
                              Board<Pixel<float>> &current_frame,
                              Board<Pixel<float>> &target_frame, bool use_gpu);
    std::string get_rule_string();
};

#endif //ZOOM_H
