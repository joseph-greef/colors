
#include "hodge.cuh"


__device__ int get_next_value_healthy(int x, int y, int *board, 
                                      int death_threshold, int k1, int k2,
                                      int width, int height) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width) % width;
            check_y = (j + height) % height;
            offset = check_y * width + check_x;

            ill += board[offset] == death_threshold;
            infected += board[offset] > 0 &&
                        board[offset] < death_threshold;
        }
    }
    return (infected / k1) + (ill / k2);
}

__device__ int get_next_value_infected(int x, int y, int *board, 
                                       int death_threshold, int infection_rate,
                                       int width, int height) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0, sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width) % width;
            check_y = (j + height) % height;
            offset = check_y * width + check_x;

            ill += board[offset] == death_threshold;
            infected += board[offset] > 0 &&
                        board[offset] < death_threshold;
            if(board[offset] > 0) {
                sum += board[offset];
            }
        }
    }
    return sum / (ill + infected + 1) + infection_rate;
}

__device__ int get_sum_neighbors(int x, int y, int *board, 
                                 int width, int height) {
    int check_x = 0, check_y = 0, offset = 0;
    int sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width) % width;
            check_y = (j + height) % height;
            offset = check_y * width + check_x;

            if(board[offset] > 0) {
                sum += board[offset];
            }
        }
    }
    return sum;
}

__global__ void cuda_hodge(int* board, int* board_buffer, int death_threshold,
                           int infection_rate, int infection_threshold,
                           int width, int height) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        
        int j = index / width;
        int i = index % width;

        if(board[index] <= 0) {
            board_buffer[index] = (int)(get_sum_neighbors(i, j, board,
                                                          width, height) >= 
                                        infection_threshold);
        }
        else if(board[index] < death_threshold) {
            board_buffer[index] = get_sum_neighbors(i, j, board, width, height) / 
                                   9;
            board_buffer[index] += infection_rate;
        }
        else if(board[index] >= death_threshold) {
            board_buffer[index] = 0;
        }

        index += blockDim.x * gridDim.x;
    }
}

__global__ void cuda_hodgepodge(int* board, int* board_buffer, int death_threshold,
                                int infection_rate, int k1, int k2,
                                int width, int height) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        
        int j = index / width;
        int i = index % width;

        if(board[index] <= 0) {
            board_buffer[index] = get_next_value_healthy(i, j, board, 
                                                         death_threshold, k1, k2,
                                                         width, height);
        }
        else if(board[index] < death_threshold) {
            board_buffer[index] = get_next_value_infected(i, j, board,
                                                          death_threshold,
                                                          infection_rate, 
                                                          width, height);
        }
        else if(board[index] >= death_threshold) {
            board_buffer[index] = 0;
        }

        index += blockDim.x * gridDim.x;
    }
}

void call_cuda_hodge(int *board, int *board_buffer, int death_threshold,
                     int infection_rate, int infection_threshold, 
                     int width, int height) {
    cuda_hodge<<<512, 128>>>(board, board_buffer, death_threshold,
                             infection_rate, infection_threshold,
                             width, height);
}

void call_cuda_hodgepodge(int *board, int *board_buffer, int death_threshold,
                     int infection_rate, int k1, int k2, 
                     int width, int height) {
    cuda_hodgepodge<<<512, 128>>>(board, board_buffer, death_threshold,
                                  infection_rate, k1, k2,
                                  width, height);
}

