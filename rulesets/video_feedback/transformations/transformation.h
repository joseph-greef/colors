
#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <cstdint>
#include <random>

union Pixel {
    struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    } part;
    uint32_t value;

};

static_assert(sizeof(Pixel) == sizeof(uint32_t));

class Transformation {
protected:
    int height_;
    int width_;

    std::random_device rd_;
    std::mt19937 e2_;
    std::uniform_real_distribution<> dist_full_;
    std::uniform_real_distribution<> dist_positive_;

    Pixel interpolate_pixel(float x, float y, Pixel *buffer);
public:
    Transformation(int width, int height);
    virtual ~Transformation();

    //should write every pixel of target_frame
    virtual void apply_transformation(Pixel *last_frame, Pixel *current_frame,
                                      Pixel *target_frame, bool use_gpu) = 0;
};

Pixel interpolate(float x, float y, Pixel *buffer);

#endif //TRANSFORMATION_H
