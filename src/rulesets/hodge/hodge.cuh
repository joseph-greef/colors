
#ifndef _HODGE_CUH
#define _HODGE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "buffer.cuh"


void call_hodge_kernel(Buffer<int> *board, Buffer<int> *board_buffer,
                       int death_threshold,
                       int infection_rate, int infection_threshold);

void call_hodgepodge_kernel(Buffer<int> *board, Buffer<int> *board_buffer,
                            int death_threshold,
                            int infection_rate, int k1, int k2);

__host__ __device__
int get_next_value_healthy(int x, int y, Buffer<int> *board, int death_threshold,
                           int k1, int k2);

__host__ __device__
int get_next_value_infected(int x, int y, Buffer<int> *board, int death_threshold,
                            int infection_rate);

__host__ __device__
int get_sum_neighbors(int x, int y, Buffer<int> *board);

__host__ __device__
void hodge_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                int death_threshold, int infection_rate, int infection_threshold);

__host__ __device__
void hodgepodge_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                     int death_threshold, int infection_rate, int k1, int k2);

#endif //_HODGE_CUH

