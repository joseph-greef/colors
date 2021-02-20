
#ifndef BOARD_CUH
#define BOARD_CUH

#include "cuda_runtime.h"

template <class T>
class Board {
private:

public:
    Board(int width, int height, bool gpu);
    ~Board();

    void copy_board_from(Board<T> &other);
    __host__ __device__ inline T get(int x, int y) {
        return data_[y * width_ + x];
    }
    __host__ __device__ inline void set(int x, int y, T value) {
        data_[y * width_ + x] = value;
    }

    int width_;
    int height_;
    bool gpu_;
    T *data_;
};

#include "board.h.cu"

#endif //BOARD_CUH

