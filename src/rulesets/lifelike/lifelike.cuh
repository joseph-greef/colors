
#ifndef _LIFELIKE_CUH
#define _LIFELIKE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void call_lifelike_kernel(int *board, int *board_buffer, bool *born,
                          bool *stay_alive, int num_faders, int current_tick,
                          int width, int height);

__host__ __device__
    void lifelike_step(int* board, int* board_buffer, int index,
                       bool *born, bool *stay_alive,
                       int num_faders, int current_tick, int width, int height);

#endif //_LIFELIKE_CUH
