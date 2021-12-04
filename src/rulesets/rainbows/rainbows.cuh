
#ifndef _RAINBOWS_CUH
#define _RAINBOWS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "board.cuh"


void call_rainbows_kernel(Board<int> *board, Board<int> *board_buffer);


#endif //_RAINBOWS_CUH
