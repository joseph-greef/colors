
#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <cstdint>
#include <iomanip>
#include <random>
#include <string>

#include "board.cuh"
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
    virtual void apply_transformation(Board<Pixel<float>> &last_frame,
                                      Board<Pixel<float>> &current_frame,
                                      Board<Pixel<float>> &target_frame, bool use_gpu) = 0;
    virtual std::string get_rule_string() = 0;
};

#endif //TRANSFORMATION_H
