
#include <algorithm>
#include <typeinfo>

template <class T>
Board<T>::Board(int width, int height)
    : width_(width)
    , height_(height)
    , device_copy_(NULL)
{
    cudaMalloc((void**)&device_data_, width_ * height_ * sizeof(T));
    host_data_ = new T[width_*height_];
    host_data_alloced_ = true;

    cudaMalloc((void**)&device_copy_, sizeof(Board<T>));
    cudaMemcpy(device_copy_, this, sizeof(Board<T>),
               cudaMemcpyHostToDevice);
}

template <class T>
Board<T>::Board(int width, int height, T *host_data)
    : width_(width)
    , height_(height)
    , host_data_(host_data)
    , device_copy_(NULL)
{
    cudaMalloc((void**)&device_data_, width_ * height_ * sizeof(T));
    host_data_alloced_ = false;

    cudaMalloc((void**)&device_copy_, sizeof(Board<T>));
    cudaMemcpy(device_copy_, this, sizeof(Board<T>),
               cudaMemcpyHostToDevice);
}

template <class T>
Board<T>::~Board() {
    cudaFree((void*)device_data_);
    cudaFree((void*)device_copy_);
    if(host_data_alloced_) {
        delete [] host_data_;
    }
}

template <class T>
void Board<T>::clear() {
    memset(host_data_, 0, width_*height_*sizeof(T));
    cudaMemset(device_data_, 0, width_*height_*sizeof(T));
}

template <class T>
void Board<T>::copy_device_to_host() {
    cudaMemcpy(host_data_, device_data_,
               width_ * height_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template <class T>
void Board<T>::copy_host_to_device() {
    cudaMemcpy(device_data_, host_data_, width_ * height_ * sizeof(T),
               cudaMemcpyHostToDevice);
}

template <class T>
void Board<T>::copy_from_board(Board<T> *other, bool use_gpu) {
    cudaMemcpy(host_data_, other->get_data(use_gpu), width_ * height_ * sizeof(T),
               cudaMemcpyHostToHost);
    if(use_gpu) {
        copy_host_to_device();
    }
}

template <class T>
T* Board<T>::get_data(bool gpu) {
    if(gpu) {
        cudaMemcpy(host_data_, device_data_,
                   width_ * height_ * sizeof(T), cudaMemcpyDeviceToHost);
    }
    return host_data_;
}

template <class T>
std::size_t Board<T>::get_type() {
    return typeid(T).hash_code();
}

template <class T>
void Board<T>::set_host_data(T *new_host_data, int new_width, int new_height) {
    if(host_data_alloced_) {
        delete [] host_data_;
    }


    //Dimensions changed
    if(new_width > 0 || new_height > 0) {
        width_ = std::max(new_width, width_);
        height_ = std::max(new_height, height_);

        cudaFree((void*)device_data_);
        cudaMalloc((void**)&device_data_, width_ * height_ * sizeof(T));
    }
    if(new_host_data == NULL) {
        host_data_ = new T[width_*height_];
        host_data_alloced_ = true;
    }
    else {
        host_data_ = new_host_data;
        host_data_alloced_ = false;
    }
}

