
#include "hodge.cuh"


__host__ __device__
int get_next_value_healthy(int x, int y, Buffer<int> *board, int death_threshold,
                           int k1, int k2) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + board->width_) % board->width_;
            check_y = (j + board->height_) % board->height_;
            offset = check_y * board->width_ + check_x;

            ill += board->get(offset) == death_threshold;
            infected += board->get(offset) > 0 &&
                        board->get(offset) < death_threshold;
        }
    }
    return (infected / k1) + (ill / k2);
}

__host__ __device__
int get_next_value_infected(int x, int y, Buffer<int> *board,
                            int death_threshold, int infection_rate) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0, sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + board->width_) % board->width_;
            check_y = (j + board->height_) % board->height_;
            offset = check_y * board->width_ + check_x;

            ill += board->get(offset) == death_threshold;
            infected += board->get(offset) > 0 &&
                        board->get(offset) < death_threshold;
            if(board->get(offset) > 0) {
                sum += board->get(offset);
            }
        }
    }
    return sum / (ill + infected + 1) + infection_rate;
}

__host__ __device__
int get_sum_neighbors(int x, int y, Buffer<int> *board) {
    int check_x = 0, check_y = 0, offset = 0;
    int sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + board->width_) % board->width_;
            check_y = (j + board->height_) % board->height_;
            offset = check_y * board->width_ + check_x;

            if(board->get(offset) > 0) {
                sum += board->get(offset);
            }
        }
    }
    return sum;
}

__host__ __device__
void hodge_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                int death_threshold, int infection_rate, int infection_threshold) {
    int j = index / board->width_;
    int i = index % board->width_;

    if(board->get(index) <= 0) {
        board_buffer->set(index, (int)(get_sum_neighbors(i, j, board) >=
                                    infection_threshold));
    }
    else if(board->get(index) < death_threshold) {
        board_buffer->set(index, get_sum_neighbors(i, j, board) / 9 + infection_rate);
    }
    else if(board->get(index) >= death_threshold) {
        board_buffer->set(index, 0);
    }
}

__host__ __device__
void hodgepodge_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                     int death_threshold, int infection_rate, int k1, int k2) {
    int j = index / board->width_;
    int i = index % board->width_;

    if(board->get(index) <= 0) {
        board_buffer->set(index, get_next_value_healthy(i, j, board,
                                                     death_threshold, k1, k2));
    }
    else if(board->get(index) < death_threshold) {
        board_buffer->set(index, get_next_value_infected(i, j, board,
                                                      death_threshold,
                                                      infection_rate));
    }
    else if(board->get(index) >= death_threshold) {
        board_buffer->set(index, 0);
    }
}

__global__
void hodge_kernel(Buffer<int>* board, Buffer<int>* board_buffer, int death_threshold,
                  int infection_rate, int infection_threshold) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < board->height_ * board->width_) {
        hodge_step(board, board_buffer, index, death_threshold, infection_rate,
                   infection_threshold);
        index += blockDim.x * gridDim.x;
    }
}

__global__
void hodgepodge_kernel(Buffer<int>* board, Buffer<int>* board_buffer,
                       int death_threshold, int infection_rate, int k1, int k2) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < board->height_ * board->width_) {
        hodgepodge_step(board, board_buffer, index, death_threshold,
                        infection_rate, k1, k2);
        index += blockDim.x * gridDim.x;
    }
}

void call_hodge_kernel(Buffer<int> *board, Buffer<int> *board_buffer,
                       int death_threshold, int infection_rate,
                       int infection_threshold) {
    hodge_kernel<<<512, 128>>>(board->device_copy_, board_buffer->device_copy_,
                               death_threshold,
                               infection_rate, infection_threshold);
}

void call_hodgepodge_kernel(Buffer<int> *board, Buffer<int> *board_buffer,
                            int death_threshold, int infection_rate,
                            int k1, int k2) {
    hodgepodge_kernel<<<512, 128>>>(board->device_copy_, board_buffer->device_copy_,
                                    death_threshold,
                                    infection_rate, k1, k2);
}

