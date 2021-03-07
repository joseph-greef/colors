
template <class T>
Board<T>::Board(int width, int height, bool gpu)
    : width_(width)
    , height_(height)
    , gpu_(gpu)
{
    if(gpu_) {
        cudaMalloc((void**)&data_, width_ * height_ * sizeof(T));
    }
    else {
        data_ = new T[width_*height_];
    }
}

template <class T>
Board<T>::~Board() {
    if(gpu_) {
        cudaFree((void*)data_);
    }
    else {
        delete [] data_;
    }
}

template <class T>
void Board<T>::copy_board_from(Board<T> &other) {
    if(gpu_ && other.gpu_) {
        cudaMemcpy(data_, other.data_, width_ * height_ * sizeof(T),
                   cudaMemcpyDeviceToDevice);
    }
    else if(!gpu_ && other.gpu_) {
        cudaMemcpy(data_, other.data_, width_ * height_ * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }
    else if(gpu_ && !other.gpu_) {
        cudaMemcpy(data_, other.data_, width_ * height_ * sizeof(T),
                   cudaMemcpyHostToDevice);
    }
    else if(!gpu_ && !other.gpu_) {
        cudaMemcpy(data_, other.data_, width_ * height_ * sizeof(T),
                   cudaMemcpyHostToHost);
    }
}

