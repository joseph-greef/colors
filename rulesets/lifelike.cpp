
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
    , initializer_(width, height)
    , num_faders_(0)
    , rainbows_(width, height)
{
    //Random pretty pattern
    //bool born_tmp[9] = {0, 0, 0, 0, 1, 1 ,1 ,1, 1};
    //bool stay_alive_tmp[9] = {1, 0, 0, 1, 1, 1 ,1 ,1, 0};

    //star wars
    bool born_tmp[9] = {0, 0, 1, 0, 0, 0 ,0 ,0, 0};
    bool stay_alive_tmp[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
    num_faders_ = 4;

    memcpy(born_, born_tmp, sizeof(born_));
    memcpy(stay_alive_, stay_alive_tmp, sizeof(stay_alive_));

    board_ = new int[width*height];
    board_buffer_ = new int[width*height];

    initializer_.init_center_cross(board_);

    InputManager::add_var_changer(&num_faders_,         SDLK_a, 1, 0, INT_MAX, "Num Faders");
}

LifeLike::~LifeLike() {
    delete board_;
    delete board_buffer_;
}

void LifeLike::copy_board_to_gpu() {
#ifdef USE_GPU
    cudaMemcpy(cudev_board_, board_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
#endif //USE_GPU
}

void LifeLike::copy_rules_to_gpu() {
#ifdef USE_GPU
    cudaMemcpy(cudev_born_, born_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cudev_stay_alive_, stay_alive_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
#endif //USE_GPU
}

void LifeLike::free_cuda() {
#ifdef USE_GPU
    std::cout << "Stopping CUDA" << std::endl;
    if(cudev_board_) {
        cudaFree((void*)cudev_board_);
    }
    if(cudev_board_buffer_) {
        cudaFree((void*)cudev_board_buffer_);
    }
    if(cudev_born_) {
        cudaFree((void*)cudev_born_);
    }
    if(cudev_stay_alive_) {
        cudaFree((void*)cudev_stay_alive_);
    }
    cudaDeviceSynchronize();

    cudev_board_ = NULL;
    cudev_board_buffer_ = NULL;
    cudev_born_ = NULL;
    cudev_stay_alive_ = NULL;
#endif //USE_GPU
}

void LifeLike::get_pixels(uint32_t *pixels) {
    rainbows_.age_to_pixels(board_, pixels);
}

void LifeLike::handle_input(SDL_Event event, bool control, bool shift) {
    bool board_changed = false;
    
    rainbows_.handle_input(event, control, shift);

    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_e:
                initializer_.init_center_square(board_);
                board_changed = true;
                break;
            case SDLK_i:
                initializer_.init_board(board_);
                board_changed = true;
                break;
            case SDLK_r:
                randomize_ruleset();
                break;
            case SDLK_w:
                initializer_.init_center_diamond(board_);
                board_changed = true;
                break;
            case SDLK_x:
                initializer_.init_center_cross(board_);
                board_changed = true;
                break;

         }
    }
    else if(event.type == SDL_KEYUP) {
        switch(event.key.keysym.sym) {
        }
    }

    if(board_changed) {
#ifdef USE_GPU
        if(use_gpu_) {
            copy_board_to_gpu();
        }
#endif
    }
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

void LifeLike::randomize_ruleset() {
    for(int i = 0; i < 9; i++) {
        born_[i] = (rand()%100>20 ? 1 : 0);
        stay_alive_[i] = (rand()%100>20 ? 1 : 0);
    }
    born_[0] = false;

    rainbows_.randomize_colors();

#ifdef USE_GPU
    if(use_gpu_) {
        copy_rules_to_gpu();
    }
#endif //USE_GPU
}

void LifeLike::setup_cuda() {
#ifdef USE_GPU
    std::cout << "Starting CUDA" << std::endl;
    cudaMalloc((void**)&cudev_board_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_born_, 9 * sizeof(bool));
    cudaMalloc((void**)&cudev_stay_alive_, 9 * sizeof(bool));

    copy_board_to_gpu();
    copy_rules_to_gpu();

    cudaDeviceSynchronize();

#endif //USE_GPU
}

void LifeLike::tick() {
    if(use_gpu_) {
#if USE_GPU
        int *temp = NULL;

        call_cuda_lifelike(cudev_board_, cudev_board_buffer_, cudev_born_,
                           cudev_stay_alive_, num_faders_, width_, height_);

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
                    board_buffer_[offset] = board_[offset] + 1;
                }
                else {
                    board_buffer_[offset] = -1;
                }

            }
            else if(board_[offset] <= -num_faders_ || board_[offset] == 0) {
                if(born_[neighbors]) {
                    board_buffer_[offset] = 1;
                }
                else if(board_[offset] == 0) {
                    board_buffer_[offset] = 0;
                }
                else {
                    board_buffer_[offset] = board_[offset] - 1;
                }
            }
            else {
                board_buffer_[offset] = board_[offset] - 1;
            }
        }
    }

    {
        int *tmp = board_buffer_;
        board_buffer_ = board_;
        board_ = tmp;
    }

}

