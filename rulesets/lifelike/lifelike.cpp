
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
    , random_fader_modulo_(6)
{
    //Random pretty pattern
    //bool born_tmp[9] = {0, 0, 0, 0, 1, 1 ,1 ,1, 1};
    //bool stay_alive_tmp[9] = {1, 0, 0, 1, 1, 1 ,1 ,1, 0};

    //star wars
    //bool born_tmp[9] = {0, 0, 1, 0, 0, 0 ,0 ,0, 0};
    //bool stay_alive_tmp[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
    //num_faders_ = 4;

    //replicator
    bool born_tmp[9] = {0, 1, 0, 1, 0, 1 ,0 ,1, 0};
    bool stay_alive_tmp[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
    num_faders_ = 0;

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

    initializer_.init_center_cross();
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

std::string LifeLike::get_name() {
    return "LifeLike";
}

void LifeLike::get_pixels(uint32_t *pixels) {
    rainbows_.age_to_pixels(board_, pixels);
}

std::string LifeLike::get_rule_string() {
    std::ostringstream rule_ss;
    for(int i = 0; i < 9; i++) {
        rule_ss << born_[i] << " ";
    }
    for(int i = 0; i < 9; i++) {
        rule_ss << stay_alive_[i] << " ";
    }
    rule_ss << num_faders_;
    return rule_ss.str();
}

void LifeLike::load_rule_string(std::string rules) {
    std::istringstream rule_ss(rules);
    for(int i = 0; i < 9; i++) {
        rule_ss >> born_[i];
    }
    for(int i = 0; i < 9; i++) {
        rule_ss >> stay_alive_[i];
    }
    rule_ss >> num_faders_;
}

void LifeLike::print_human_readable_rules() {
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
    std::cout << "Num Faders: " << num_faders_ << std::endl;
}

void LifeLike::randomize_ruleset() {
    for(int i = 0; i < 9; i++) {
        born_[i] = (rand()%100>20 ? 1 : 0);
        stay_alive_[i] = (rand()%100>20 ? 1 : 0);
    }
    born_[0] = false;

    num_faders_ = (rand() % random_fader_modulo_);

    rainbows_.randomize_colors();

#ifdef USE_GPU
    if(use_gpu_) {
        copy_rules_to_gpu();
    }
#endif //USE_GPU
}

void LifeLike::start() { 
    std::cout << "Starting LifeLike" << std::endl;
    Ruleset::start();

    ADD_FUNCTION_CALLER(&LifeLike::randomize_ruleset, SDL_SCANCODE_R, false, false,
                        "LifeLike", "Randomize ruleset");

    InputManager::add_int_changer(&num_faders_, SDL_SCANCODE_A,
                                  false, false, 0, INT_MAX,
                                  "LifeLike", "Number of refractory states after death");
    InputManager::add_int_changer(&random_fader_modulo_, SDL_SCANCODE_A,
                                  true, false, 0, INT_MAX,
                                  "LifeLike", "Set modulo for number of refractory states during randomization");

    initializer_.start();
    rainbows_.start();
}

void LifeLike::stop() { 
    Ruleset::stop();

    InputManager::remove_var_changer(SDL_SCANCODE_R, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_A, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_A, true, false);

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


