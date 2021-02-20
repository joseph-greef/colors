
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
    void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                              Pixel *target_frame, bool use_gpu);
};

#endif //ZOOM_H
