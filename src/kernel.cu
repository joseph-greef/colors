#include "kernel.cuh"

#define lowdivergence 0
#define reallylowdivergence 0


__device__ int get_num_alive_neighbors(int x, int y, int neighborhood_type, int range, int *board, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    int check_x, check_y, count = 0;

    if (neighborhood_type == VON_NEUMANN) {
        for (int i = x - range; i <= x + range; i++) {
            for (int j = y - range; j <= y + range; j++) {
                if (j == y && i == x)
                    continue;
                if (abs(i - x) + abs(j - y) <= range) {


                    check_x = (i + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;
                    check_y = (j + CUDA_CELL_HEIGHT) % CUDA_CELL_HEIGHT;
                    //and check the coordinate, if it's alive increase count
                    if (board[check_y*CUDA_CELL_WIDTH + check_x] > 0)
                        count++;
                }
            }
        }
    }
    else {
        for (int i = x - range; i <= x + range; i++) {
            for (int j = y - range; j <= y + range; j++) {
                if (j == y && i == x)
                    continue;


                check_x = (i + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;
                check_y = (j + CUDA_CELL_HEIGHT) % CUDA_CELL_HEIGHT;
                //and check the coordinate, if it's alive increase count
                if (board[check_y*CUDA_CELL_WIDTH + check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}



__device__ int get_sum_neighbors(int x, int y, int neighborhood_type, int range, int * board, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    int check_x, check_y, count = 0;

    if(neighborhood_type == VON_NEUMANN) {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                if(abs(i-x)+abs(j-y) <= range) {


                    check_x = (i + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;
                    check_y = (j + CUDA_CELL_HEIGHT) % CUDA_CELL_HEIGHT;
                    count += board[check_y*CUDA_CELL_WIDTH + check_x];
                }
            }
        }
    }
    else {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                check_x = (i + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;
                check_y = (j + CUDA_CELL_HEIGHT) % CUDA_CELL_HEIGHT;
                count += board[check_y*CUDA_CELL_WIDTH + check_x];
            }
        }
    }
    return count;

}

//used for the 1D update algorithm
__global__ void cuda_move_board_up(int *board, int *board_buffer, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + CUDA_CELL_WIDTH;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {
        int board_val = board[index];
        board_buffer[index - CUDA_CELL_WIDTH] = board_val + (board_val > 0) - (board_val < 0);
        index += blockDim.x * gridDim.x;
    }
}

//updates the board with the 1D rules
__global__ void cuda_update_board_1D(int *board, int *board_buffer, int *OneD_rules, int OneD_width, int alive_offset, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = CUDA_CELL_HEIGHT - 2;
    while (i < CUDA_CELL_WIDTH) {

        int power = 1;
        int sum = 0;
        int high_bound = i + OneD_width/2;
        int low_bound = i - OneD_width/2;

        //adds the sum with each neighbor being abinary digit
        for (int check_i = high_bound; check_i >= low_bound; check_i--) {
            int check_x = (check_i + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;
            if (board[j * CUDA_CELL_WIDTH + check_x] <= 0) {
                sum += power;
            }
            power *= 2;
        }
        //check the rules and set the next state accordingly
        if (OneD_rules[sum]) {
            board_buffer[(j+1)*CUDA_CELL_WIDTH + i] = alive_offset;
        }
        else {
            board_buffer[(j+1)*CUDA_CELL_WIDTH + i] = 0;
        }
        i += blockDim.x * gridDim.x;
    }
}

void call_cuda_UB1D(int *board, int *board_buffer, int *OneD_rules, int OneD_width, int alive_offset, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_move_board_up<<<512, 128>>>(board, board_buffer, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
    cuda_update_board_1D<<<4, 128>>>(board, board_buffer, OneD_rules, OneD_width, alive_offset, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}


//update the board for hodge rules
__global__ void cuda_update_board_hodge(int* board, int* board_buffer, int * hodge_rules, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {
        int j = index / CUDA_CELL_WIDTH;
        int i = index % CUDA_CELL_WIDTH;

        //if healthy
        if (board[index] == 0) {
            if (get_sum_neighbors(i, j, hodge_rules[4], hodge_rules[5], board, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT) > 1) {
                board_buffer[index] = 1;
            }

        }
        //infected
        else if (board[index] < hodge_rules[3]) {
            int sum = get_sum_neighbors(i, j, hodge_rules[4], hodge_rules[5], board, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
            board_buffer[index] = (sum / 9) + hodge_rules[2];
            if (board_buffer[index] > hodge_rules[3]) {
                board_buffer[index] = hodge_rules[3];
            }
        }
        //ill
        else {
            board_buffer[index] = 0;
        }

        index += blockDim.x * gridDim.x;
    }

}

void call_cuda_UBH(int *board, int *board_buffer, int *hodge_rules, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_board_hodge<<<512, 128>>>(board, board_buffer, hodge_rules, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}


//update the board with nondeterministic rules
__global__ void cuda_update_board_non_deterministic(int* board, int* board_buffer, float *born, float *stay_alive, float *rand_nums, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {

        int j = index / CUDA_CELL_WIDTH;
        int i = index % CUDA_CELL_WIDTH;

        //get how many alive neighbors it has
        int neighbors = get_num_alive_neighbors(i, j, MOORE, 1, board, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);

#if reallylowdivergence
        //translate the if statement into a single boolean expression
        int bornresult = (int) (born[neighbors]>rand_nums[index]);
        int stayresult = (int) (stay_alive[neighbors]>rand_nums[index]);
        int isdead = (int)(board[index] <= 0);
        board_buffer[index] = isdead*(bornresult + !bornresult * (board[index] - 1)) + !isdead * (-(!stayresult) + stayresult * (board[index] + 1));
#else
        if (board[index] <= 0) {
#if lowdivergence
            int result = (int) (born[neighbors]>rand_nums[index]);
            board_buffer[index] = result + !result * (board[index] - 1);
#else
            if (born[neighbors] >= rand_nums[index]) {
                board_buffer[index] = 1;
            }
            else if (board[index] < 0)
                board_buffer[index] = board[index] - 1;
#endif /*lowdivergence*/
        }
        else {
#if lowdivergence
            int result = (int) (stay_alive[neighbors]>rand_nums[index]);
            board_buffer[index] = -(!result) + result * (board[index] + 1);
#else
            if (stay_alive[neighbors] >= rand_nums[index])
                board_buffer[index] = board[index] + 1;
            else
                board_buffer[index] = -1;
#endif /*lowdivergence*/

        }
#endif /*reallylowdivergence*/


        index += blockDim.x * gridDim.x;
    }
}

void call_cuda_UBND(int *board, int *board_buffer, float *born, float *stay_alive, float *rand_nums, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_board_non_deterministic<<<512, 128>>>(board, board_buffer, born, stay_alive, rand_nums, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}


//update the board with normal rules. Also includes faders since normal lifelike is just faders with no refractory states
__global__ void cuda_update_board_normal(int* board, int* board_buffer, float *born, float *stay_alive, int num_faders, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {

        int j = index / CUDA_CELL_WIDTH;
        int i = index % CUDA_CELL_WIDTH;

        //get how many alive neighbors it has
        int neighbors = get_num_alive_neighbors(i, j, MOORE, 1, board, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);

#if reallylowdivergence
        //translate the divergent if statement into a longass boolean expression.
        int bornresult = (int) (born[neighbors]>0.5);
        int stayresult = (int) (stay_alive[neighbors]>0.5);
        int isdead = (int)(board[index] <= -num_faders);
        int isfader = (int) (board[index] > -num_faders && board[index] < 0);
        board_buffer[index] = !isfader * (isdead*(bornresult + !bornresult * (board[index] - 1)) + !isdead * (-(!stayresult) + stayresult * (board[index] + 1))) + isfader * (board[index] - 1);
#else
        if (board[index] <= -num_faders) {
#if lowdivergence
            int result = (int) (born[neighbors]>0.5);
            board_buffer[index] = result + !result * (board[index] - 1);
#else
            if (born[neighbors] >= 0.5) {
                board_buffer[index] = 1;
            }
            else if (board[index] < 0)
                board_buffer[index] = board[index] - 1;
#endif /*lowdivergence*/
        }
        else if (board[index] > 0) {
#if lowdivergence
            int result = (int) (stay_alive[neighbors]>0.5);
            board_buffer[index] = -(!result) + result * (board[index] + 1);
#else
            if (stay_alive[neighbors] >= 0.5)
                board_buffer[index] = board[index] + 1;
            else
                board_buffer[index] = -1;
#endif /*lowdivergence*/

        }
        else {
            board_buffer[index] = board[index] - 1;
        }

#endif /*reallylowdivergence*/


        index += blockDim.x * gridDim.x;
    }
}

void call_cuda_UBN(int *board, int *board_buffer, float *born, float *stay_alive, int num_faders, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_board_normal<<<512, 128>>>(board, board_buffer, born, stay_alive, num_faders, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}


//update the board accordsing to the larger than life ruleset
__global__ void cuda_update_board_LtL(int *board, int *board_buffer, int *LtL_rules, float random_num, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {

        int j = index / CUDA_CELL_WIDTH;
        int i = index % CUDA_CELL_WIDTH;

        int neighbors = get_num_alive_neighbors(i, j, LtL_rules[5], LtL_rules[0], board, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
        //if dead
#if reallylowdivergence
        //convert the if statement into a medium-ass boolean expression
        int deadresult = (neighbors >= LtL_rules[3] && neighbors <= LtL_rules[4]);
        int liveresult = (neighbors >= LtL_rules[1] && neighbors <= LtL_rules[2]);
        int isdead = board[index] <= 0;
        board_buffer[index] = isdead * ((!deadresult*(board[index] - 1)) + deadresult) + !isdead * ((liveresult*(board[index] + 1)) - !liveresult);
#else
        if (board[index] <= 0) {
#if lowdivergence
            int result = (neighbors >= LtL_rules[3] && neighbors <= LtL_rules[4]);
            board_buffer[index] = (!result*(board[index] - 1)) + result;
#else

            //if supposed to be born
            if (neighbors >= LtL_rules[3] && neighbors <= LtL_rules[4])
                // then make it alive with an age of 1
                board_buffer[index] = 1;
            //if its still dead
            //and it's aging
            else if (board[index] < 0)
                //then age it
                board_buffer[index] = board[index] - 1;
#endif /*lowdivergence*/
        }
        //if alive
        else {
#if lowdivergence
            int result = (neighbors >= LtL_rules[1] && neighbors <= LtL_rules[2]);
            board_buffer[index] = (result*(board[index] + 1)) - !result;
#else

            //and it's supposed to stay alive, and it's not older than the max age
            if (neighbors >= LtL_rules[1] && neighbors <= LtL_rules[2])
                //then age it
                board_buffer[index] = board[index] + 1;
            else
                //otherwise kill it
                board_buffer[index] = -1;
#endif /*lowdivergence*/


        }
#endif /*reallylowdivergence*/
        index += blockDim.x * gridDim.x;
    }

}

void call_cuda_UBLtL(int *board, int *board_buffer, int *LtL_rules, float random_num, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_board_LtL<<<512, 128>>>(board, board_buffer, LtL_rules, random_num, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}

//sigmoid functions for smooth life
__device__ float s1(float x, float a, float alpha, float *smooth_rules) {
    float to_return = 1 / (1 + exp(((a - x) * 4) / alpha));
    return to_return;
}
__device__ float s2(float x, float a, float b, float *smooth_rules) {
    float to_return = s1(x, a, smooth_rules[2], smooth_rules) * (1 - s1(x, b, smooth_rules[2], smooth_rules));
    return to_return;
}
__device__ float sm(float x, float y, float m, float *smooth_rules) {
    float to_return = x * (1 - s1(m, 0.5, smooth_rules[3], smooth_rules)) + y * s1(m, 0.5, smooth_rules[3], smooth_rules);
    return to_return;
}
__device__ float s(float n, float m, float *smooth_rules) {
    float to_return = s2(n, sm(smooth_rules[4], smooth_rules[6], m, smooth_rules), sm(smooth_rules[5], smooth_rules[7], m, smooth_rules), smooth_rules);
    return to_return;
}

//update the board according to the smooth rules. Pass in all the squares of the cell/neighborhood sizes so they
//are only computed once total per frame.
__global__ void cuda_update_board_smooth(float *board_float, float *board_buffer_float, float *smooth_rules,
                                            float r_a_2, float r_i_2, float r_a_2_m_b, float r_i_2_m_b, float r_a_2_p_b, float r_i_2_p_b,
                                            int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    float n, m, r, rr;
    int x, y, test_x, test_y;


    //iterate over every cell*/
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {

        int j = index / CUDA_CELL_WIDTH;
        int i = index % CUDA_CELL_WIDTH;
        n = 0;
        m = 0;
        for (y = -smooth_rules[1]; y <= smooth_rules[1]; y++) {
            test_y = (j + y + CUDA_CELL_HEIGHT) % CUDA_CELL_HEIGHT;

            for (x = -smooth_rules[1]; x <= smooth_rules[1]; x++) {
                test_x = (i + x + CUDA_CELL_WIDTH) % CUDA_CELL_WIDTH;



                rr = x*x + y*y;
                r = sqrt(rr);


                //if inside r_i
                if (rr < r_i_2_m_b)
                    n += board_float[test_y*CUDA_CELL_WIDTH + test_x];
                //inner antialiasing zone
                else if (rr < r_i_2_p_b) {
                    n += (smooth_rules[0] + 0.5 - r) * board_float[test_y*CUDA_CELL_WIDTH + test_x];
                    m += (smooth_rules[0] + 0.5 - r) * board_float[test_y*CUDA_CELL_WIDTH + test_x];
                }
                //neighborhood
                else if (rr > r_i_2_p_b && rr < r_a_2_m_b)
                    m += board_float[test_y*CUDA_CELL_WIDTH + test_x];
                //outer antialiasing zone
                else if (rr > r_a_2_m_b && rr < r_a_2_p_b)
                    m += (smooth_rules[1] + 0.5 - r) * board_float[test_y*CUDA_CELL_WIDTH + test_x];

            }
        }


        //get the normalized integrals
        n = n / (r_i_2 * PI);
        m = m / (PI * (r_a_2 - r_i_2));

        //calculate s
        float new_val = s(n, m, smooth_rules);

        //and apply the transition
        board_buffer_float[index] = board_float[index] + DT*(new_val-board_float[index]);

        index += blockDim.x * gridDim.x;

    }

}

void call_cuda_UBS(float *board, float *board_buffer, float *smooth_rules, float r_a_2, float r_i_2, float r_a_2_m_b, float r_i_2_m_b, float r_a_2_p_b, float r_i_2_p_b, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_board_smooth<<<512, 128>>>(board, board_buffer, smooth_rules, r_a_2, r_i_2, r_a_2_m_b, r_i_2_m_b, r_a_2_p_b, r_i_2_p_b, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}


//updates the int board according to the float board to make the color version work. Checks a threshold of live vs dead
//and updates according to that
__global__ void cuda_update_smooth_colors(float *board_float, int *board, int *board_buffer, float death_cutoff, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    //iterate over every cell*/
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < CUDA_CELL_HEIGHT * CUDA_CELL_WIDTH) {
#if reallylowdivergence
        //more translation of if statements into boolean expressions. Woop. Woop. Woop.
        int cellalive = (int)(board_float[index] > death_cutoff);
        int isdead = (int)(board[index] <= 0);
        board_buffer[index] = cellalive * (!isdead * (board[index] + 1) + isdead) + !cellalive * (-1*(!isdead) + isdead * (board[index] - 1));
#else
        if (board_float[index] > death_cutoff) {
#if lowdivergence
            int isdead = (int)(board[index] <= 0);
            board_buffer[index] = !isdead * (board[index] + 1) + isdead;
#else
            //increase age if alive
            if(board[index] > 0)
                board_buffer[index] = board[index] + 1;
            //and increase age if dead and supposed to change
            else if(board[index] < 0)
                board_buffer[index] = 1;
#endif /*lowdivergence*/
        }
        else {
#if lowdivergence
            int isdead = (int)(board[index] <= 0);
            board_buffer[index] = -1*(!isdead) + isdead * (board[index] - 1);
#else
            //increase age if alive
            if(board[index] > 0)
                board_buffer[index] = -1;
            //and increase age if dead and supposed to change
            else if(board[index] < 0)
                board_buffer[index] = board[index] - 1;
#endif /*lowdivergence*/

        }
#endif /*reallylowdivergence*/

        index += blockDim.x * gridDim.x;
    }
}

void call_cuda_USC(float *board_float, int *board, int *board_buffer, float death_cutoff, int CUDA_CELL_WIDTH, int CUDA_CELL_HEIGHT) {
    cuda_update_smooth_colors<<<512, 128>>>(board_float, board, board_buffer, death_cutoff, CUDA_CELL_WIDTH, CUDA_CELL_HEIGHT);
}