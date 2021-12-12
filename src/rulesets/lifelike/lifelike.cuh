
#ifndef _LIFELIKE_CUH
#define _LIFELIKE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "buffer.cuh"


void call_lifelike_kernel(Buffer<int> *board, Buffer<int> *board_buffer, bool *born,
                          bool *stay_alive, int num_faders, int current_tick);

__host__ __device__
void lifelike_step(Buffer<int>* board, Buffer<int>* board_buffer, int index,
                   bool *born, bool *stay_alive,
                   int num_faders, int current_tick);

#endif //_LIFELIKE_CUH
