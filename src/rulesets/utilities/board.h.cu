
#include <typeinfo>

template <class T>
Board<T>::Board(int width, int height)
    : width_(width)
    , height_(height)
{
    cudaMalloc((void**)&device_data_, width_ * height_ * sizeof(T));
    host_data_ = new T[width_*height_];

    cudaMalloc((void**)&device_copy_, sizeof(Board<T>));
    cudaMemcpy(device_copy_, this, sizeof(Board<T>),
               cudaMemcpyHostToDevice);
}

template <class T>
Board<T>::~Board() {
    cudaFree((void*)device_data_);
    cudaFree((void*)device_copy_);
    delete [] host_data_;
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

