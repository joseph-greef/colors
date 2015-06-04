
#include "board.h"

Board::Board() : gen(8), e2(time(NULL)), dist(0, 1) {
    board = (int*)malloc(sizeof(int) * CELL_WIDTH * CELL_HEIGHT);
    board_buffer = (int*)malloc(sizeof(int) * CELL_WIDTH * CELL_HEIGHT);
    board_float = (float*)malloc(sizeof(float) * CELL_WIDTH * CELL_HEIGHT);
    board_buffer_float = (float*)malloc(sizeof(float) * CELL_WIDTH * CELL_HEIGHT);
    born = (float*)malloc(sizeof(float) * 18);
    stay_alive = (born+9);

    OneD_rules = (int*)malloc(sizeof(int) * 8);
    OneD_width = 3;

    LtL_rules = (int*) malloc(sizeof(int) * 6);
    smooth_rules = (float*)malloc(sizeof(float) * 8); //{r_i, r_a, alpha_n, alpha_m, b1, b2, d1, d2}
    smooth_rules[0] = R_I; 
    smooth_rules[1] = R_A;
    smooth_rules[2] = ALPHA_N;
    smooth_rules[3] = ALPHA_M;
    smooth_rules[4] = B1;
    smooth_rules[5] = B2;
    smooth_rules[6] = D1;
    smooth_rules[7] = D2;

    hodge_rules = (int*)malloc(sizeof(int) * 6);

    changing_background = true;
    pause = false;
    use_gpu = false;



    cudaMalloc((void**)&dev_board, CELL_HEIGHT * CELL_WIDTH * sizeof(int));
    cudaMalloc((void**)&dev_board_buffer, CELL_HEIGHT * CELL_WIDTH * sizeof(int));

    cudaMalloc((void**)&dev_rand_nums, CELL_HEIGHT * CELL_WIDTH * sizeof(float));

    cudaMalloc((void**)&dev_board_float, CELL_HEIGHT * CELL_WIDTH * sizeof(float));
    cudaMalloc((void**)&dev_board_buffer_float, CELL_HEIGHT * CELL_WIDTH * sizeof(float));



    cudaMalloc((void**)&dev_stay_alive, 9 * sizeof(float));
    cudaMalloc((void**)&dev_born, 9 * sizeof(float));

    cudaMalloc((void**)&dev_LtL_rules, 6 * sizeof(int));
    cudaMalloc((void**)&dev_hodge_rules, 6 * sizeof(int));
    
    cudaMalloc((void**)&dev_smooth_rules, 8 * sizeof(float));

    cudaMalloc((void**)&dev_OneD_rules, 8 * sizeof(int));

    curandCreateGenerator(&rand_gen,CURAND_RNG_PSEUDO_DEFAULT);

    initialize_rules();
    update_algorithm = 0;

}


Board::~Board() {
    free(board);
    free(board_buffer);
    free(born);
}






//this function updates the board according to the ruleset
void Board::update_board() {
    switch(update_algorithm) {
        case 0:
            update_board_normal();
            break;
        case 1:
            update_board_smooth();
            break;
        case 2:
            update_board_LtL();
            break;
        case 3:
            update_board_non_deterministic();
            break;
        case 4:
            update_board_hodge();
            break;
        case 5:
            update_board_1D();
            break;
    }

    //switch the boards
    int* temp = board;
    board = board_buffer;
    board_buffer = temp;


    //if we're using GPU switch the GPU ones too
    if (use_gpu) {
        cudaMemcpy(board_buffer, dev_board_buffer, CELL_WIDTH * CELL_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

        temp = dev_board_buffer;
        dev_board_buffer = dev_board;
        dev_board = temp;

        float* temp2 = board_float;
        board_float = board_buffer_float;
        board_buffer_float = temp2;
    }
}

//gets the sum of the neighbors within range with either a VN or moore neighborhood
int Board::get_sum_neighbors(int x, int y, int neighborhood_type, int range) {
        int check_x, check_y, count = 0;

    if(neighborhood_type == VON_NEUMANN) {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                if(abs(i-x)+abs(j-y) <= range) {


                    check_x = (i + CELL_WIDTH) % CELL_WIDTH;
                    check_y = (j + CELL_HEIGHT) % CELL_HEIGHT;
                    count += board[check_y*CELL_WIDTH + check_x];
                }
            }
        }
    }
    else {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                check_x = (i + CELL_WIDTH) % CELL_WIDTH;
                check_y = (j + CELL_HEIGHT) % CELL_HEIGHT;
                count += board[check_y*CELL_WIDTH + check_x];
            }
        }
    }
    return count;

}

//updates the hodge board
void Board::update_board_hodge() {
    if (pause) {
        return;
    }

    if (use_gpu) {
        cudaMemcpy(dev_hodge_rules, hodge_rules, 6 * sizeof(int), cudaMemcpyHostToDevice);
        call_cuda_UBH(dev_board, dev_board_buffer, dev_hodge_rules);
        return;
    }

    for (int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            //if we're alive
            if (board[j*CELL_WIDTH + i] == 0) {
                if (get_sum_neighbors(i, j, hodge_rules[4], hodge_rules[5]) > 1) {
                    board_buffer[j*CELL_WIDTH + i] = 1;
                }
                //int infected = get_num_infected_neighbors(i, j, hodge_rules[4], hodge_rules[5]);
                //int ill = get_num_ill_neighbors(i, j, hodge_rules[4], hodge_rules[5]);
                //board_buffer[j*CELL_WIDTH + i] = infected / hodge_rules[0] + ill / hodge_rules[1];

            }
            //infected
            else if (board[j*CELL_WIDTH + i] < hodge_rules[3]) {
                int sum = get_sum_neighbors(i, j, hodge_rules[4], hodge_rules[5]);
                //int infected = get_num_infected_neighbors(i, j, hodge_rules[4], hodge_rules[5]);
                //int ill = get_num_ill_neighbors(i, j, hodge_rules[4], hodge_rules[5]);
                board_buffer[j*CELL_WIDTH + i] = (sum / 9) + hodge_rules[2];
                if (board_buffer[j*CELL_WIDTH + i] > hodge_rules[3]) {
                    board_buffer[j*CELL_WIDTH + i] = hodge_rules[3];
                }
                //board_buffer[j*CELL_WIDTH + i] = sum / infected + ill + 1 + hodge_rules[2];

            }
            //ill
            else {
                board_buffer[j*CELL_WIDTH+i] = 0;
            }
        }
    }

}

// moves the entire board up one cell. Removes the top row
void Board::move_board_up() {
    for (int j = 1; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) { 
            int board_val = board[j*CELL_WIDTH + i];
            board_buffer[(j - 1)*CELL_WIDTH + i] = board_val + (board_val > 0) - (board_val < 0);
        }
    }
}

//updates the board according to the 1D rules
void Board::update_board_1D() {
    static int alive_offset = 1;
    static int alive_offset_delay = 0;
    
    if (pause) {
        update_colors();
        return;
    }


    //this moves the colors down the rows
    alive_offset_delay = (alive_offset_delay + 1) % 4;
    if (alive_offset_delay == 0) {
        alive_offset++;
    }
    

    if (use_gpu) {
        cudaMemcpy(dev_OneD_rules, OneD_rules, (int)exp2(OneD_width) * sizeof(int), cudaMemcpyHostToDevice);
        call_cuda_UB1D(dev_board, dev_board_buffer, dev_OneD_rules, OneD_width, alive_offset);
        return;
    }


    move_board_up();

    int j = CELL_HEIGHT - 2;
    for (int i = 0; i < CELL_WIDTH; i++) {      
        int power = 1;
        int sum = 0;
        //sum the cells where each neighbor cell is a binary digit in the sum
        for (int check_i = i + OneD_width / 2; check_i >= i - OneD_width / 2; check_i--) {
            int check_x = (check_i + CELL_WIDTH) % CELL_WIDTH;
            if (board[j * CELL_WIDTH + check_x] <= 0) {
                sum += power;
            }
            power *= 2;
        }
        //check the rules and update accordingly  
        if (OneD_rules[sum]) {
            board_buffer[(j+1)*CELL_WIDTH + i] = alive_offset;
        }
        else {
            board_buffer[(j+1)*CELL_WIDTH + i] = 0;
        }
    }
    

}


//update the board following the Larger than Life ruleset
void Board::update_board_LtL() {
    //if we're paused, just update the colors
    if (pause) {
        update_colors();
        return;
    }

    if (use_gpu) {
        cudaMemcpy(dev_LtL_rules, LtL_rules, 6 * sizeof(int), cudaMemcpyHostToDevice);
        call_cuda_UBLtL(dev_board, dev_board_buffer, dev_LtL_rules, 0.5);
        return;
    }

    //if a cell is positive it's alive, if it's negative or 0 then it's dead.
    //the actual value is how many generations it has been alive.
    //if the value is 0 then that dead cell is not supposed to age
    //this is how the non-changing background is implemented

    //iterate over every cell
    for(int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            int neighbors = get_num_alive_neighbors(i, j, LtL_rules[5], LtL_rules[0]);
            //if dead
            if(board[j*CELL_WIDTH+i] <= 0) {
                //if supposed to be born
                if(neighbors >= LtL_rules[3] && neighbors <= LtL_rules[4])
                    // then make it alive with an age of 1
                    board_buffer[j*CELL_WIDTH+i] = 1;
                //if its still dead
                //and it's aging
                else if(board[j*CELL_WIDTH+i] < 0)
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] - 1;
                else
                    //keep it dead
                    board_buffer[j*CELL_WIDTH+i] = 0;
            }
            //if alive
            else {
                //and it's supposed to stay alive, and it's not older than the max age
                if(neighbors >= LtL_rules[1] && neighbors <= LtL_rules[2])
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] + 1;
                else
                    //otherwise kill it
                    board_buffer[j*CELL_WIDTH+i] = -1;

            }
        }
    }


}


//gets the number of live neighbors that the cell at x,y has withing a neighborhood of range
int Board::get_num_alive_neighbors(int x, int y, int neighborhood_type, int range) {
    int check_x, check_y, count = 0;

    if(neighborhood_type == VON_NEUMANN) {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                if(j==y && i==x)
                    continue;
                if(abs(i-x)+abs(j-y) <= range) {


                    check_x = (i + CELL_WIDTH) % CELL_WIDTH;
                    check_y = (j + CELL_HEIGHT) % CELL_HEIGHT;
                    //and check the coordinate, if it's alive increase count
                    if(board[check_y*CELL_WIDTH+check_x] > 0)
                        count++;
                }
            }
        }
    }
    else {
        for(int i = x - range; i <= x + range; i++) {
            for(int j = y - range; j <= y + range; j++) {
                if(j==y && i==x)
                    continue;


                check_x = (i + CELL_WIDTH) % CELL_WIDTH;
                check_y = (j + CELL_HEIGHT) % CELL_HEIGHT;
                //and check the coordinate, if it's alive increase count
                if(board[check_y*CELL_WIDTH+check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}

//updates the boards according to the smooth life rules
void Board::update_board_smooth() {

    float n, m, r, rr;
    int x, y, test_x, test_y, num_m, num_n;
    float r_a_2 = smooth_rules[1] * smooth_rules[1];
    float r_i_2 = smooth_rules[0] * smooth_rules[0];
    //squares (m)inus b
    float r_a_2_m_b = (smooth_rules[1]-0.5)*(smooth_rules[1]-0.5);
    float r_i_2_m_b = (smooth_rules[0]-0.5)*(smooth_rules[0]-0.5);
    //squares (p)lus b
    float r_a_2_p_b = (smooth_rules[1]+0.5)*(smooth_rules[1]+0.5);
    float r_i_2_p_b = (smooth_rules[0]+0.5)*(smooth_rules[0]+0.5);

    if (use_gpu) {
        cudaMemcpy(dev_smooth_rules, smooth_rules, 8 * sizeof(float), cudaMemcpyHostToDevice);
        call_cuda_UBS(dev_board_float, dev_board_buffer_float, dev_smooth_rules, r_a_2, r_i_2, r_a_2_m_b, r_i_2_m_b, r_a_2_p_b, r_i_2_p_b);
        call_cuda_USC(dev_board_buffer_float, dev_board, dev_board_buffer, 0.5);
        cudaMemcpy(board_buffer_float, dev_board_buffer_float, CELL_WIDTH * CELL_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
        float *temp = dev_board_buffer_float;
        dev_board_buffer_float = dev_board_float;
        dev_board_float = temp;
        return;
    }

    
    //iterate over every cell
    for(int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            n = 0;
            m = 0;
            //check every pixel in the square of side length 2*R_a
            for (y = -smooth_rules[1]; y <= smooth_rules[1]; y++) {
                test_y = (j + y + CELL_HEIGHT) % CELL_HEIGHT;

                for (x = -smooth_rules[1]; x <= smooth_rules[1]; x++) {
                    test_x = (i + x + CELL_WIDTH) % CELL_WIDTH;

                    //figure out how far away from the center it is
                    rr = x*x + y*y;
                    r = sqrt(rr);
                
                    //if inside r_i
                    if (rr < r_i_2_m_b)
                        n += board_float[test_y*CELL_WIDTH + test_x];
                    //if it's in the anti aliasing zone between r_i and r_a
                    else if (rr < r_i_2_p_b) {
                        n += (smooth_rules[0] + 0.5 - r) * board_float[test_y*CELL_WIDTH + test_x];
                        m += (smooth_rules[0] + 0.5 - r) * board_float[test_y*CELL_WIDTH + test_x];
                    }
                    //if it's only in r_a
                    else if (rr >= r_i_2_p_b && rr < r_a_2_m_b)
                        m += board_float[test_y*CELL_WIDTH + test_x];
                    //if it's in the outer antialiasing zone 
                    else if (rr >= r_a_2_m_b && rr < r_a_2_p_b)
                        m += (smooth_rules[1] + 0.5 - r) * board_float[test_y*CELL_WIDTH + test_x];

                }
            }


            //get the normalized integral values
            n = n / (r_i_2 * PI);
            m = m / (PI * (r_a_2 - r_i_2));
       
            //and the next value
            float new_val = s(n, m);
        
            //and move DX of the way to that new value
            board_buffer_float[j*CELL_WIDTH+i] = board_float[j*CELL_WIDTH+i] + DT*(new_val-board_float[j*CELL_WIDTH+i]);
        }
    }
    

}

//sigmoid functions for smooth life
float Board::s1(float x, float a, float alpha) {
    float to_return = 1/(1 + exp(((a-x) * 4)/alpha));
    return to_return;
}
float Board::s2(float x, float a, float b) {
    float to_return = s1(x, a, smooth_rules[2]) * (1 - s1(x, b, smooth_rules[2]));
    return to_return;
}
float Board::sm(float x, float y, float m) {
    float to_return = x * (1 - s1(m, 0.5, smooth_rules[3])) + y * s1(m, 0.5, smooth_rules[3]);
    return to_return;
}
float Board::s(float n, float m) {
    float to_return = s2(n, sm(smooth_rules[4], smooth_rules[6], m), sm(smooth_rules[5], smooth_rules[7], m));
    return to_return;
}



void Board::update_board_non_deterministic() {
    //if we're paused, just update the colors
    if(pause) {
        update_colors();
        return;
    }

    if (use_gpu) {
        cudaMemcpy(dev_born, born, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_stay_alive, stay_alive, 9 * sizeof(float), cudaMemcpyHostToDevice);
        curandGenerateUniform(rand_gen, dev_rand_nums, CELL_HEIGHT * CELL_WIDTH);
        //cudaMemcpy(dev_board, board, CELL_WIDTH * CELL_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
        call_cuda_UBND(dev_board, dev_board_buffer, dev_born, dev_stay_alive, dev_rand_nums);
        return;
    }

    //if a cell is positive it's alive, if it's negative or 0 then it's dead.
    //the actual value is how many generations it has been alive.
    //if the value is 0 then that dead cell is not supposed to age
    //this is how the non-changing background is implemented

    //iterate over every cell
    for(int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            //get how many alive neighbors it has
            int neighbors = get_num_alive_neighbors(i, j, MOORE, 1);

            //if dead
            if(board[j*CELL_WIDTH+i] <= 0) {
                //if supposed to be born
                if(born[neighbors] >= dist(e2))
                    // then make it alive with an age of 1
                    board_buffer[j*CELL_WIDTH+i] = 1;
                //if its still dead
                //and it's aging
                else if(board[j*CELL_WIDTH+i] < 0)
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] - 1;
                else
                    //keep it dead
                    board_buffer[j*CELL_WIDTH+i] = 0;
            }
            //if alive
            else {
                //and it's supposed to stay alive, and it's not older than the max age
                if(stay_alive[neighbors] >= dist(e2))
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] + 1;
                else
                    //otherwise kill it
                    board_buffer[j*CELL_WIDTH+i] = -1;

            }
        }
    }
}


void Board::update_board_normal() {
    //if we're paused, just update the colors
    if(pause) {
        update_colors();
        return;
    }

    if (use_gpu) {
        cudaMemcpy(dev_born, born, 9 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_stay_alive, stay_alive, 9 * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(dev_board, board, CELL_WIDTH * CELL_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
        call_cuda_UBN(dev_board, dev_board_buffer, dev_born, dev_stay_alive, num_faders);
        return;
    }

    //if a cell is positive it's alive, if it's negative or 0 then it's dead.
    //the actual value is how many generations it has been alive.
    //if the value is 0 then that dead cell is not supposed to age
    //this is how the non-changing background is implemented

    //iterate over every cell
    for(int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            //get how many alive neighbors it has
            int neighbors = get_num_alive_neighbors(i, j, MOORE, 1);

            //if dead
            if(board[j*CELL_WIDTH+i] <= -num_faders) {
                //if supposed to be born
                if(born[neighbors] >= 0.5)
                    // then make it alive with an age of 1
                    board_buffer[j*CELL_WIDTH+i] = 1;
                //if its still dead
                //and it's aging
                else if(board[j*CELL_WIDTH+i] < 0)
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] - 1;
                else
                    //keep it dead
                    board_buffer[j*CELL_WIDTH+i] = 0;
            }
            //if alive
            else if (board[j*CELL_WIDTH+i] > 0){
                //and it's supposed to stay alive, and it's not older than the max age
                if(stay_alive[neighbors] >= 0.5)
                    //then age it
                    board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] + 1;
                else
                    //otherwise kill it
                    board_buffer[j*CELL_WIDTH+i] = -1;

            }
            else {
                board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] - 1;
            }
        }
    }


}




//this function just updates the colors without updating the board
void Board::update_colors() {
    //iterate over the whole board
    for(int j = 0; j < CELL_HEIGHT; j++) {
        for(int i = 0; i < CELL_WIDTH; i++) {
            //increase age if alive
            if(board[j*CELL_WIDTH+i] > 0)
                board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] + 1;
            //and increase age if dead and supposed to change
            else if(board[j*CELL_WIDTH+i] < 0)
                board_buffer[j*CELL_WIDTH+i] = board[j*CELL_WIDTH+i] - 1;
        }
    }
}


void Board::initialize_rules() {
    //initial rules for the automata is game of life
    //int life_born[9] =       {0,1,1,1,1,0,0,1,0};
    //int life_stay_alive[9] = {0,1,1,0,0,1,1,1,1};
    
    //int life_born[9] =       {0,1,0,0,0,0,0,0,0};
    //int life_stay_alive[9] = {0,0,0,0,0,0,0,0,0};

    //copier
    //int life_born[9] =       {0,1,0,1,0,1,0,1,0};
    //int life_stay_alive[9] = {0,1,0,1,0,1,0,1,0};
    //vote
    //int life_born[9] =       {0,0,0,0,1,0,1,1,1};
    //int life_stay_alive[9] = {0,0,0,1,0,1,1,1,1};
    //float life_born[9] =       {-.0224982,-0.0439246,0.16508,-0.381114,0.0156361,0.254988,0.354988,0.2501,0.31466};
    //float life_stay_alive[9] = {1,1,1,1,1,1,1,1,1};

    int life_born[9] =       {0,0,1,0,1,1,1,1,1};
    int life_stay_alive[9] = {1,0,0,1,0,1,1,1,1};
    

    free(born);

    born = (float*)malloc(sizeof(float) * 18);
    stay_alive = (born + 9);


    for(int i = 0; i < 9; i++) {
        born[i] = life_born[i];
        stay_alive[i] = life_stay_alive[i];
    }
    
    num_faders = 7;

    //bugs int ltl[6] = {5, 34, 58, 34, 45, MOORE};
    int ltl[6] = {7, 113, 225, 113, 225, MOORE};
    for(int i = 0; i < 6; i++) {
        LtL_rules[i] = ltl[i];
    }

    int hodge[6] = {1, 1, 1, 128, MOORE, 4};
    for(int i = 0; i < 6; i++) {
        hodge_rules[i] = hodge[i];
    }

    OneD_width = 3;
    free(OneD_rules);
    cudaFree(dev_OneD_rules);
    cudaMalloc((void**)&dev_OneD_rules, (int)exp2(3) * sizeof(int));
    OneD_rules = (int*)malloc(sizeof(int) * (int)exp2(3));

    int one_rules[8] = {0, 1, 0, 1, 1, 0, 1, 0};
    for (int i = 0; i < 8; i++) {
        OneD_rules[i] = one_rules[i];
    }

}

// 0 1 1 1 1 0 0 1 0
// 1 1 1 0 0 1 1 1 1c


int* Board::get_board() {
    return board;
}

float* Board::get_board_float() {
    return board_float;
}

//randomizes all the rules scept smooth and non-deterministic
void Board::randomize_rules() {
    for(int i = 0; i < 9; i++) {
        born[i] = (rand()%100>20 ? 1 : 0);
        stay_alive[i] = (rand()%100>20 ? 1 : 0);
    }

    cudaMemcpy(dev_born, born, 9, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_stay_alive, stay_alive, 9, cudaMemcpyHostToDevice);

    LtL_rules[0] = rand() % 10;
    LtL_rules[2] = rand() % (LtL_rules[0]+1) * (LtL_rules[0]+1);
    LtL_rules[1] = LtL_rules[2] - rand() % (LtL_rules[0]+1) * (LtL_rules[0]+1);
    LtL_rules[4] = rand() % (LtL_rules[0]+1) * (LtL_rules[0]+1);
    LtL_rules[3] = LtL_rules[2] - rand() % (LtL_rules[0]+1) * (LtL_rules[0]+1);
    LtL_rules[5] = rand() % 2;

    OneD_width = (rand() % 3) * 2 + 3;
    free(OneD_rules);
    cudaFree(dev_OneD_rules);
    cudaMalloc((void**)&dev_OneD_rules, (int)exp2(OneD_width) * sizeof(int));
    OneD_rules = (int*)malloc(sizeof(int) * (int)exp2(OneD_width));
    for (int i = 0; i < (int)exp2(OneD_width); i++) {
        OneD_rules[i] = (rand()%100>20 ? 1 : 0);
    }

}

void Board::randomize_rules_non_deterministic() {
    std::normal_distribution<> ndist(0.0, 0.3);
    for(int i = 0; i < 9; i++) {
        born[i] = (rand()%100>20 ? ndist(e2) : 0);
        stay_alive[i] = (rand()%100>10 ? 1 : 0);
    }
}

void Board::randomize_rules_smooth() {
    std::normal_distribution<> ndist(0, 0.1);
   /* smooth_rules[0] = (ndist(e2) + 1) * 15;
    smooth_rules[1] = smooth_rules[0] / ((ndist(e2) + 0.5) * 3);
    smooth_rules[2] = ndist(e2) / 5;
    smooth_rules[3] = ndist(e2) / 5;
    smooth_rules[4] = ndist(e2)/2 * 0.1;
    smooth_rules[5] = smooth_rules[4] + ndist(e2) / 3;
    smooth_rules[6] = ndist(e2)/2 * 0.1;
    smooth_rules[7] = smooth_rules[6] + ndist(e2) / 3;*/

    smooth_rules[2] = smooth_rules[2] + ndist(e2);
    smooth_rules[3] = smooth_rules[3] + ndist(e2);
    smooth_rules[4] = smooth_rules[4] + ndist(e2);
    smooth_rules[5] = smooth_rules[5] + ndist(e2);
    smooth_rules[6] = smooth_rules[6] + ndist(e2);
    smooth_rules[7] = smooth_rules[7] + ndist(e2);
}


void Board::set_update_algorithm(int new_algorithm) {
    update_algorithm = new_algorithm;
}



bool Board::get_changing_background() {
    return changing_background;
}

void Board::toggle_changing_background() {
    changing_background = !changing_background;
}

//switch to or from the GPU
void Board::toggle_use_gpu() {
    use_gpu = !use_gpu;
    //if we're switching to the GPU put the board onto the GPU
    if (use_gpu) {
        send_board_to_GPU();
    }
}
void Board::toggle_pause() {
    pause = !pause;
}


//this is genetic algorithm stuff. Used to either seed or not seed
//the algorithm with a given ruleset. Only works for lifelike and nondeterminstic life
void Board::rules_pretty() {
    gen.add_seed(born);
    born = gen.generate_one_mean();
    stay_alive = (born + 9);
    //init_center_dot();
}

void Board::rules_not_pretty() {
    free(born);
    born = gen.generate_one_mean();
    stay_alive = (born + 9);
    //init_center_dot();
}

void Board::rules_pretty_float() {
    gen.add_seed(born);
    born = gen.generate_one_mean_float();
    stay_alive = (born + 9);
    //init_center_dot();
}

void Board::rules_not_pretty_float() {
    free(born);
    born = gen.generate_one_mean_float();
    stay_alive = (born + 9);
    //init_center_dot();
}

void Board::print_rules() {
    for (int i = 0; i < 9; i++) {
        std::cout << born[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << stay_alive[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 6; i++) {
        std::cout << LtL_rules[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << smooth_rules[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;



}

int * Board::get_hodge_rules() {
    return hodge_rules;
}


void Board::send_board_to_GPU() {
    cudaMemcpy(dev_board, board, CELL_HEIGHT * CELL_WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_board_float, board_float, CELL_HEIGHT * CELL_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

}

void Board::modify_num_faders(int factor) {
    num_faders += factor;
    if (num_faders < 0) 
        num_faders = 0;
}