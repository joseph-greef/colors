
template <class T>
__host__ __device__ Pixel<T> interpolate(float x, float y, Board<Pixel<T>> &board) {
    if(x >= board.width_ || x < 0 || y >= board.height_ || y < 0) {
        Pixel<T> p = {0, 0, 0, 0};
        return p;
    }
    int x1 = std::floor(x);
    int x2 = std::ceil(x);
    int y1 = std::floor(y);
    int y2 = std::ceil(y);
    float x_bias = x - x1;
    float y_bias = y - y1;

    T br_bias = 1024 * x_bias * y_bias;
    T bl_bias = 1024 * (1-x_bias) * y_bias;
    T tr_bias = 1024 * x_bias * (1-y_bias);
    T tl_bias = 1024 - br_bias - bl_bias - tr_bias;

    Pixel<T> tl = board.get(x1, y1); //buffer[y1 * width + x1].value;
    Pixel<T> tr = board.get(x2, y1); //buffer[y1 * width + x2].value;
    Pixel<T> bl = board.get(x2, y1); //buffer[y2 * width + x1].value;
    Pixel<T> br = board.get(x2, y1); //buffer[y2 * width + x2].value;

    Pixel<T> dest = {0, 0, 0, 0};

    for(int i = 0; i < 4; i++) {
        T temp = tl.value[i] * tl_bias + 
                 tr.value[i] * tr_bias +
                 bl.value[i] * bl_bias +
                 br.value[i] * br_bias;

        temp /= 1024;

        if(temp > 255) {
            temp = 255;
        }
        else if(temp < 0) {
            temp = 0;
        }

        dest.value[i] = temp;
    }

    return dest;
}

template <class T>
__host__ __device__ T truncate(T value) {
    if(value > 255) {
        return 255;
    }
    else if(value < 0) {
        return 0;
    }
    else {
        return value;
    }
}
