
#include <cmath>
#include <iostream>

#include "transformation.h"


Transformation::Transformation(int width, int height)
    : height_(height)
    , width_(width)
    , e2_(rd_())
    , dist_full_(-1, 1)
    , dist_positive_(0, 1)
{
}

Pixel Transformation::interpolate_pixel(float x, float y, Pixel *buffer) {
    //assert(x > 0);
    //assert(y > 0);
    if(x >= width_ || x < 0 || y >= height_ || y < 0) {
        Pixel p;
        p.value = 0;
        return p;
    }
    int x1 = std::floor(x);
    int x2 = std::ceil(x);
    int y1 = std::floor(y);
    int y2 = std::ceil(y);
    float x_bias = x - x1;
    float y_bias = y - y1;

    int32_t br_bias = 1024 * x_bias * y_bias;
    int32_t bl_bias = 1024 * (1-x_bias) * y_bias;
    int32_t tr_bias = 1024 * x_bias * (1-y_bias);
    int32_t tl_bias = 1024 - br_bias - bl_bias - tr_bias;

    uint32_t tl = buffer[y1 * width_ + x1].value;
    uint32_t tr = buffer[y1 * width_ + x2].value;
    uint32_t bl = buffer[y2 * width_ + x1].value;
    uint32_t br = buffer[y2 * width_ + x2].value;

    uint8_t channels[4] = { 0 };

    for(int i = 0; i < 4; i++) {
        int32_t temp = (tl & 0xFF) * tl_bias + 
                       (tr & 0xFF) * tr_bias +
                       (bl & 0xFF) * bl_bias +
                       (br & 0xFF) * br_bias;

        temp /= 1024;

        if(temp > 255) {
            std::cout << temp << std::endl;
            temp = 255;
        }
        else if(temp < 0) {
            std::cout << temp << std::endl;
            temp = 0;
        }

        channels[i] = temp;

        tl = tl >> 8;
        tr = tr >> 8;
        bl = bl >> 8;
        br = br >> 8;
    }

    return *(Pixel*)channels;
}

Transformation::~Transformation() {
}

