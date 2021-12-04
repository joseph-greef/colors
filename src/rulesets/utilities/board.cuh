
#ifndef BOARD_CUH
#define BOARD_CUH

#include "cuda_runtime.h"



template <class T>
class Board {
private:
    T *device_data_;
    T *host_data_;
public:
    Board<T> *device_copy_;
    int width_;
    int height_;

    Board(int width, int height);
    ~Board();

    void clear();
    void copy_host_to_device();

    T* get_data(bool gpu);
    std::size_t get_type();

    __host__ __device__ inline T get(int x, int y) {
        return get(y * width_ + x);
    }

    __host__ __device__ inline T get(int index) {
#ifdef __CUDA_ARCH__
        return device_data_[index];
#else
        return host_data_[index];
#endif //__CUDA_ARCH__
    }
    __host__ __device__ inline void set(int x, int y, T value) {
        return set(y * width_ + x, value);
    }

    __host__ __device__ inline void set(int index, T value) {
#ifdef __CUDA_ARCH__
        device_data_[index] = value;
#else
        host_data_[index] = value;
#endif //__CUDA_ARCH__
    }

};

#include "board.h.cu"

#endif //BOARD_CUH

