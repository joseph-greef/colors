
#ifndef BUFFER_CUH
#define BUFFER_CUH

#include <cstdint>

#include "cuda_runtime.h"

#define NOT_COMPATIBLE ((size_t)0)
#define RGBA_BUFFER ((size_t)1)
#define INT_BUFFER (typeid(int).hash_code())

template <class T>
union Pixel {
    struct {
        T b;
        T g;
        T r;
        T a;
    } part;
    T value[4];
};
static_assert(sizeof(Pixel<uint8_t>) == sizeof(uint32_t));

__host__ __device__ inline Pixel<uint8_t> uint32_to_pixel(uint32_t color) {
    Pixel<uint8_t> p = { 0 };
    p.part.b = static_cast<uint8_t>((color >> 0) & 0xFF);
    p.part.g = static_cast<uint8_t>((color >> 8) & 0xFF);
    p.part.r = static_cast<uint8_t>((color >> 16) & 0xFF);
    p.part.a = static_cast<uint8_t>((color >> 24) & 0xFF);
    return p;
}

template <class T>
class Buffer {
private:
    T *device_data_;
    T *host_data_;
    bool host_data_alloced_;
public:
    Buffer<T> *device_copy_;
    int w_;
    int h_;

    Buffer(int width, int height);
    Buffer(int width, int height, T *host_data);
    ~Buffer();

    void clear();
    void copy_device_to_host();
    void copy_host_to_device();
    void copy_from_buffer(Buffer<T> *other, bool gpu);

    T* get_data(bool gpu);
    std::size_t get_type();
    void set_host_data(T *new_host_data, int width, int height);

    __host__ __device__ inline T get(int x, int y) {
        return get(y * w_ + x);
    }

    __host__ __device__ inline T get(int index) {
#ifdef __CUDA_ARCH__
        return device_data_[index];
#else
        return host_data_[index];
#endif //__CUDA_ARCH__
    }
    __host__ __device__ inline void set(int x, int y, T value) {
        return set(y * w_ + x, value);
    }

    __host__ __device__ inline void set(int index, T value) {
#ifdef __CUDA_ARCH__
        device_data_[index] = value;
#else
        host_data_[index] = value;
#endif //__CUDA_ARCH__
    }
};

#include "buffer.h.cu"

#endif //BUFFER_CUH

