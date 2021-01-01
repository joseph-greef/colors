
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "board_constants.h"

#include <SDL2/SDL.h>
#include <iostream>
#include <stdio.h>

#define DT 0.1

void call_cuda_UB1D(int *board, int *board_buffer, int *OneD_rules, int OneD_width, int alive_offset, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_UBN(int *board, int *board_buffer, float *born, float *stay_alive, int num_faders, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_UBND(int *board, int *board_buffer, float *born, float *stay_alive, float *, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_UBLtL(int *board, int *board_buffer, int *LtL_rules, float random_num, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_UBS(float *board, float *board_buffer, float *smooth_rules, float r_a_2, float r_i_2, float r_a_2_m_b, float r_i_2_m_b, float r_a_2_p_b, float r_i_2_p_b, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_USC(float *board_float, int *board, int *board_buffer, float death_cutoff, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
void call_cuda_UBH(int *board, int *board_buffer, int *hodge_rules, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT);
