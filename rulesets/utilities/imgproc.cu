
#include "imgproc.cuh"

__host__ __device__ Pixel interpolate(float x, float y, int width, int height, Pixel *buffer) {
    if(x >= width || x < 0 || y >= height || y < 0) {
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

    uint32_t tl = buffer[y1 * width + x1].value;
    uint32_t tr = buffer[y1 * width + x2].value;
    uint32_t bl = buffer[y2 * width + x1].value;
    uint32_t br = buffer[y2 * width + x2].value;

    uint8_t channels[4] = { 0 };

    for(int i = 0; i < 4; i++) {
        int32_t temp = (tl & 0xFF) * tl_bias + 
                       (tr & 0xFF) * tr_bias +
                       (bl & 0xFF) * bl_bias +
                       (br & 0xFF) * br_bias;

        temp /= 1024;

        if(temp > 255) {
            temp = 255;
        }
        else if(temp < 0) {
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
