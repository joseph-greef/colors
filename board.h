#ifndef BOARD_H_INCLUDED
#define BOARD_H_INCLUDED


#include <stdlib.h>
#include <random>
#include <time.h>
#include <math.h>
#include <iostream>
#include <algorithm>

#include "curand.h"
#include "cuda_runtime.h"
#include "kernel.cuh"

#include <curand_precalc.h>
#include <curand_kernel.h>


#include "info.h"
#include "RuleGenerator.h"

#include "board_constants.h"



class Board {
    private:
        struct saved_rule {
            float born[9];
            float stay_alive[9];
            int ltl[6];
            int smooth[8];
            int num_faders;
        };

        saved_rule saved_rules[10];

        int *board;
        int *board_buffer;

        int *dev_board;
        int *dev_board_buffer;

        float *dev_rand_nums;

        float *board_float;
        float *board_buffer_float;

        float *dev_board_float;
        float *dev_board_buffer_float;

        float *born;
        float *stay_alive;
        int num_faders;

        float *dev_born;
        float *dev_stay_alive;


        int *LtL_rules;
        int *dev_LtL_rules;

        int *OneD_rules;
        int *dev_OneD_rules;
        int OneD_width;

        int *hodge_rules;
        int *dev_hodge_rules;
        
        float *smooth_rules;
        float *dev_smooth_rules;

        bool use_gpu;

        int CELL_WIDTH;
        int CELL_HEIGHT;

        curandGenerator_t rand_gen;
        
        int update_algorithm;
        bool changing_background;
        bool pause;
        RuleGenerator gen;

       

        std::mt19937 e2;
        std::uniform_real_distribution<float> dist;

        void update_board_hodge();
        void update_board_non_deterministic();
        void update_board_normal();
        void update_board_smooth();
        void update_board_LtL();
        void update_board_1D();

        void move_board_up();

        void free_cuda();
        void alloc_cuda();

        void free_mem();
        void alloc_mem();
        
        int get_num_alive_neighbors(int x, int y, int neighborhood_type, int range);
        int get_sum_neighbors(int x, int y, int neighborhood_type, int range);

        float s1(float x, float a, float alpha);
        float s2(float x, float a, float b);
        float sm(float x, float y, float m);
        float s(float n, float m);
    public:
        Board();
        ~Board();
        

        void update_board();

        int get_cell_width();
        int get_cell_height();

        void set_cell_width(int new_width);
        void set_cell_height(int new_height);

        void update_colors();

        int* get_board();
        float *get_board_float();

        bool get_changing_background();
        void toggle_changing_background();

        int get_faders();

        int* get_hodge_rules();



        void initialize_rules();

        void randomize_rules();
        void randomize_rules_non_deterministic();
        void randomize_rules_smooth();

        void set_update_algorithm(int new_algorithm);



        void toggle_pause();
        void toggle_use_gpu();

        void rules_pretty();
        void rules_not_pretty();

        void rules_pretty_float();
        void rules_not_pretty_float();

        void print_rules();

        void send_board_to_GPU();

        void modify_num_faders(int factor);

        void save_rules(int slot);
        void recall_rules(int slot);

        bool get_use_gpu();
};


#endif // BOARD_H_INCLUDED
