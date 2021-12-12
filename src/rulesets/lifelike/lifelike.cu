
#include "lifelike.cuh"

__host__ __device__ static
int get_num_alive_neighbors(int x, int y, Buffer<int> *board)
{

    int check_x, check_y, count = 0;
    for (int i = x - 1; i <= x + 1; i++) {
        for (int j = y - 1; j <= y + 1; j++) {
            if (j == y && i == x)
                continue;


            check_x = (i + board->width_) % board->width_;
            check_y = (j + board->height_) % board->height_;
            //and check the coordinate, if it's alive increase count
            if (board->get(check_x, check_y) > 0) {
                count++;
            }
        }
    }
    return count;

    /* A non-functional but maybe faster version waiting to be debuged.
    int top = (y + 1) % height;
    int bot = (y + height - 1) % height;
    int right = (x + 1) % width;
    int left = (x + width - 1) % width;

    int count = 0;
    count += !!board[top * width + left] + !!board[top * width + x] + !!board[top * width + right];
    count += !!board[y * width + left] +                          + !!board[y * width + right];
    count += !!board[bot * width + left] + !!board[bot * width + x] + !!board[bot * width + right];

    return count;
    */
}

__host__ __device__
void lifelike_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                   bool *born, bool *stay_alive,
                   int num_faders, int current_tick)
{
    int j = index / board->width_;
    int i = index % board->width_;
    //get how many alive neighbors it has
    int neighbors = get_num_alive_neighbors(i, j, board);

    if(board->get(index) > 0) {
        if(stay_alive[neighbors]) {
            board_buffer->set(index, board->get(index));
        }
        else {
            board_buffer->set(index, -current_tick);
        }
    }
    else if(board->get(index) + current_tick >= num_faders) {
        if(born[neighbors]) {
            board_buffer->set(index, current_tick);
        }
        else {
            board_buffer->set(index, board->get(index));
        }
    }
    else {
        board_buffer->set(index, board->get(index));
    }
}


__global__ static
void lifelike_kernel(Buffer<int>* board, Buffer<int>* board_buffer,
                     bool *born, bool *stay_alive, int num_faders,
                     int current_tick) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < board->height_ * board->width_) {

        lifelike_step(board, board_buffer, index, born, stay_alive, num_faders,
                      current_tick);


        index += blockDim.x * gridDim.x;
    }
}


void call_lifelike_kernel(Buffer<int> *board, Buffer<int> *board_buffer, bool *born,
                          bool *stay_alive, int num_faders, int current_tick) {
    lifelike_kernel<<<512, 128>>>(board->device_copy_, board_buffer->device_copy_,
                                  born, stay_alive, num_faders, current_tick);
}

