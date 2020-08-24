

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define VON_NEUMANN 0
#define MOORE 1

__device__ int get_num_alive_neighbors(int x, int y, int neighborhood_type, int range, int *board, int width, int height);
