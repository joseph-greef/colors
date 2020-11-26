
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
    , initializer_(1, 54, width, height)
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

#ifdef USE_GPU
    std::cout << "Allocating CUDA memory for LifeLike" << std::endl;
    cudaMalloc((void**)&cudev_board_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_born_, 9 * sizeof(int));
    cudaMalloc((void**)&cudev_stay_alive_, 9 * sizeof(bool));
#endif //USE_GPU

    initializer_.init_center_cross(board_);
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
#ifdef USE_GPU
    if(board_changed) {
        if(use_gpu_) {
            copy_board_to_gpu();
        }
    }
#endif
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

void LifeLike::start() { 
    InputManager::add_var_changer(&num_faders_, SDLK_a, 1, 0, INT_MAX, "Num Faders");

    initializer_.start();
    rainbows_.start();
}

void LifeLike::stop() { 
    InputManager::remove_var_changer(SDLK_a);

    initializer_.stop();
    rainbows_.stop();
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


