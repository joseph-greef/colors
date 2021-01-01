
#ifndef _HODGE_CUH
#define _HODGE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void call_cuda_hodge(int *board, int *board_buffer, int death_threshold,
                     int infection_rate, int infection_threshold, 
                     int width, int height);

void call_cuda_hodgepodge(int *board, int *board_buffer, int death_threshold,
                          int infection_rate, int k1, int k2, 
                          int width, int height);
#endif //_HODGE_CUH

