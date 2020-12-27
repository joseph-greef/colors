
#include <climits>
#include <iostream>
#include <stdlib.h>

#ifdef USE_GPU
#include "curand.h"
#include "cuda_runtime.h"
#include "lifelike.cuh"
#endif

#include "input_manager.h"
#include "lifelike.h"

LifeLike::LifeLike(int width, int height)
    : Ruleset(width, height)
    , initializer_(&board_, 1, 54, width, height)
    , num_faders_(0)
    , rainbows_(width, height, 1)
{
    //Random pretty pattern
    //bool born_tmp[9] = {0, 0, 0, 0, 1, 1 ,1 ,1, 1};
    //bool stay_alive_tmp[9] = {1, 0, 0, 1, 1, 1 ,1 ,1, 0};

    //star wars
    bool born_tmp[9] = {0, 0, 1, 0, 0, 0 ,0 ,0, 0};
    bool stay_alive_tmp[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
    num_faders_ = 4;

    //life
    //bool born_tmp[9] = {0, 0, 0, 1, 0, 0 ,0 ,0, 0};
    //bool stay_alive_tmp[9] = {0, 0, 1, 1, 0, 0, 0, 0, 0};
    //num_faders_ = 0;

    current_tick_ = num_faders_;

    memcpy(born_, born_tmp, sizeof(born_));
    memcpy(stay_alive_, stay_alive_tmp, sizeof(stay_alive_));

    board_ = new int[width*height];
    board_buffer_ = new int[width*height];

#ifdef USE_GPU
    std::cout << "Allocating CUDA memory for LifeLike" << std::endl;
    cudaMalloc((void**)&cudev_board_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_born_, 9 * sizeof(int));
    cudaMalloc((void**)&cudev_stay_alive_, 9 * sizeof(bool));
#endif //USE_GPU

    initializer_.init_center_cross(false, false);
}

LifeLike::~LifeLike() {
    delete [] board_;
    delete [] board_buffer_;
#ifdef USE_GPU
    std::cout << "Freeing CUDA memory for LifeLike" << std::endl;
    cudaFree((void*)cudev_board_);
    cudaFree((void*)cudev_board_buffer_);
    cudaFree((void*)cudev_born_);
    cudaFree((void*)cudev_stay_alive_);
#endif //USE_GPU
}

#ifdef USE_GPU
void LifeLike::copy_board_to_gpu() {
    cudaMemcpy(cudev_board_, board_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void LifeLike::copy_rules_to_gpu() {
    cudaMemcpy(cudev_born_, born_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cudev_stay_alive_, stay_alive_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void LifeLike::start_cuda() {
    copy_rules_to_gpu();
    copy_board_to_gpu();
}

void LifeLike::stop_cuda() {
}

#endif //USE_GPU

void LifeLike::get_pixels(uint32_t *pixels) {
    rainbows_.age_to_pixels(board_, pixels);
}

void LifeLike::print_controls() {
    std::cout << std::endl << "LifeLike Controls:" << std::endl;
    std::cout << "E: Initialize center square" << std::endl;
    std::cout << "I: Initialixe random board" << std::endl;
    std::cout << "R: Randomize ruleset" << std::endl;
    std::cout << "W: Initialize center diamond" << std::endl;
    std::cout << "X: Initialize center cross" << std::endl;

    rainbows_.print_rules();
}

void LifeLike::print_rules() {
    std::cout << "Born: {";
    for(int i = 0; i < 9; i++) {
        std::cout << born_[i] << ", ";
    }
    std::cout << "}" << std::endl;

    std::cout << "Stay Alive: {";
    for(int i = 0; i < 9; i++) {
        std::cout << stay_alive_[i] << ", ";
    }
    std::cout << "}" << std::endl;
}

void LifeLike::randomize_ruleset(bool control, bool shift) {
    for(int i = 0; i < 9; i++) {
        born_[i] = (rand()%100>20 ? 1 : 0);
        stay_alive_[i] = (rand()%100>20 ? 1 : 0);
    }
    born_[0] = false;

    rainbows_.randomize_colors(control, shift);

#ifdef USE_GPU
    if(use_gpu_) {
        copy_rules_to_gpu();
    }
#endif //USE_GPU
}

void LifeLike::start() { 
    std::cout << "Starting LifeLike" << std::endl;
    Ruleset::start();

    ADD_FUNCTION_CALLER(&LifeLike::randomize_ruleset, SDLK_r,
                        "(Life) Randomize Ruleset");

    InputManager::add_int_changer(&num_faders_, SDLK_a, 0, INT_MAX, "(Life) Num Faders");

    initializer_.start();
    rainbows_.start();
}

void LifeLike::stop() { 
    Ruleset::stop();

    InputManager::remove_var_changer(SDLK_r);

    InputManager::remove_var_changer(SDLK_a);

    initializer_.stop();
    rainbows_.stop();
}

void LifeLike::tick() {
    if(initializer_.was_board_changed()) {
        current_tick_ = num_faders_;
#ifdef USE_GPU
        if(use_gpu_) {
            copy_board_to_gpu();
        }
#endif
    }

    if(use_gpu_) {
#if USE_GPU
        int *temp = NULL;

        call_cuda_lifelike(cudev_board_, cudev_board_buffer_, cudev_born_,
                           cudev_stay_alive_, num_faders_, current_tick_,
                           width_, height_);

        cudaMemcpy(board_, cudev_board_buffer_, 
                   width_ * height_ * sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        temp = cudev_board_buffer_;
        cudev_board_buffer_ = cudev_board_;
        cudev_board_ = temp;
#endif //USE_GPU
    }
    else {
        update_board();
    }
    current_tick_++;
}

void LifeLike::update_board() {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            //get how many alive neighbors it has
            int neighbors = Ruleset::get_num_alive_neighbors(board_, i, j, 1,
                                                             Moore);
            int offset = j * width_ + i;

            if(board_[offset] > 0) {
                if(stay_alive_[neighbors]) {
                    board_buffer_[offset] = board_[offset];
                }
                else {
                    board_buffer_[offset] = -current_tick_;
                }

            }
            //board_ + current_tick_ is the number of ticks since state change
            //so this block is when the cell is allowed to be born
            else if(board_[offset] + current_tick_ >= num_faders_) {
                if(born_[neighbors]) {
                    board_buffer_[offset] = current_tick_;
                }
                else {
                    board_buffer_[offset] = board_[offset];
                }
            }
            //this block is the refractory states, just chill
            else {
                board_buffer_[offset] = board_[offset];
            }
        }
    }

    {
        int *tmp = board_buffer_;
        board_buffer_ = board_;
        board_ = tmp;
    }

}


