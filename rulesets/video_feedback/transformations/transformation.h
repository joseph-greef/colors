
#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <cstdint>
#include <random>

#include "imgproc.cuh"


class Transformation {
protected:
    int height_;
    int width_;

    std::random_device rd_;
    std::mt19937 e2_;
    std::uniform_real_distribution<> dist_full_;
    std::uniform_real_distribution<> dist_positive_;

public:
    Transformation(int width, int height);
    virtual ~Transformation();

    //should write every pixel of target_frame
    virtual void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame, bool use_gpu) = 0;
};

#endif //TRANSFORMATION_H
