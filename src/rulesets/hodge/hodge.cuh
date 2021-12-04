
#ifndef _HODGE_CUH
#define _HODGE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



void call_hodge_kernel(int *board, int *board_buffer, int death_threshold,
                       int infection_rate, int infection_threshold,
                       int width, int height);

void call_hodgepodge_kernel(int *board, int *board_buffer, int death_threshold,
                            int infection_rate, int k1, int k2,
                            int width, int height);

__host__ __device__ int get_next_value_healthy(int x, int y, int *board,
                                               int death_threshold,
                                               int k1, int k2,
                                               int width, int height);

__host__ __device__ int get_next_value_infected(int x, int y, int *board,
                                                int death_threshold,
                                                int infection_rate,
                                                int width, int height);

__host__ __device__ int get_sum_neighbors(int x, int y, int *board,
                                          int width, int height);

__host__ __device__ void hodge_step(int* board, int* board_buffer, int index,
                                    int death_threshold,
                                    int infection_rate, int infection_threshold,
                                    int width, int height);

__host__ __device__ void hodgepodge_step(int* board, int* board_buffer, int index,
                                         int death_threshold,
                                         int infection_rate, int k1, int k2,
                                         int width, int height);

#endif //_HODGE_CUH

