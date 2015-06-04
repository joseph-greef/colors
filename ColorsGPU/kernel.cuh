
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "info.h"
#include "board_constants.h"

#include <SDL.h>
#include <iostream>
#include <stdio.h>

void call_cuda_UB1D(int *board, int *board_buffer, int *OneD_rules, int OneD_width, int alive_offset);
void call_cuda_UBN(int *board, int *board_buffer, float *born, float *stay_alive, int num_faders);
void call_cuda_UBND(int *board, int *board_buffer, float *born, float *stay_alive, float *rand_nums);
void call_cuda_UBLtL(int *board, int *board_buffer, int *LtL_rules, float random_num);
void call_cuda_UBS(float *board, float *board_buffer, float *smooth_rules, float r_a_2, float r_i_2, float r_a_2_m_b, float r_i_2_m_b, float r_a_2_p_b, float r_i_2_p_b);
void call_cuda_USC(float *board_float, int *board, int *board_buffer, float death_cutoff);
void call_cuda_UBH(int *board, int *board_buffer, int *hodge_rules);
