
#include <climits>
#include <iostream>
#include <stdlib.h>

#undef USE_GPU
#ifdef USE_GPU
#include "curand.h"
#include "cuda_runtime.h"
#include "hodge.cuh"
#endif

#include "input_manager.h"
#include "hodge.h"

Hodge::Hodge(int width, int height)
    : Ruleset(width, height)
    , changing_background_(false)
    , death_threshold_(255)
    , infection_rate_(15)
    , infection_threshold_(2)
    , initializer_(width, height)
    , k1_(3)
    , k2_(3)
    , podge_(false)
    , rainbows_(width, height)
{
    board_ = new int[width*height];
    board_buffer_ = new int[width*height];

    InputManager::add_var_changer(&death_threshold_, SDLK_g, 25, 0, INT_MAX, "Death Threshold");
    InputManager::add_var_changer(&infection_rate_, SDLK_h, 10, INT_MIN, INT_MAX, "Infection Rate");
    InputManager::add_var_changer(&infection_threshold_, SDLK_j, 1, 0, INT_MAX, "Infection Theshold");
    InputManager::add_var_changer(&k1_, SDLK_k, 1, 0, INT_MAX, "k1");
    InputManager::add_var_changer(&k2_, SDLK_l, 1, 0, INT_MAX, "k2");

    initializer_.init_board(board_);
}

Hodge::~Hodge() {
    delete board_;
    delete board_buffer_;
}

std::string Hodge::Name = std::string("Hodge");

void Hodge::copy_board_to_gpu() {
#ifdef USE_GPU
    cudaMemcpy(cudev_board_, board_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
#endif //USE_GPU
}

void Hodge::copy_rules_to_gpu() {
    //TODO
}

void Hodge::free_cuda() {
#ifdef USE_GPU
    std::cout << "Freeing CUDA" << std::endl;
    cudaFree((void*)cudev_board_);
    cudaFree((void*)cudev_board_buffer_);
    cudaFree((void*)cudev_born_);
    cudaFree((void*)cudev_stay_alive_);
#endif //USE_GPU
}

void Hodge::get_pixels(uint32_t *pixels) {
    rainbows_.age_to_pixels(board_, pixels);
}

void Hodge::handle_input(SDL_Event event, bool control, bool shift) {
    bool board_changed = false;

    rainbows_.handle_input(event, control, shift);

    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_b:
                changing_background_ = !changing_background_;
                break;
            case SDLK_e:
                initializer_.init_center_square(board_);
                board_changed = true;
                break;
            case SDLK_i:
                initializer_.init_board(board_);
                board_changed = true;
                break;
            case SDLK_o:
                podge_ = !podge_;
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

int Hodge::get_next_value_healthy(int x, int y) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width_) % width_;
            check_y = (j + height_) % height_;
            offset = check_y * width_ + check_x;

            ill += board_[offset] == death_threshold_;
            infected += board_[offset] > 0 &&
                        board_[offset] < death_threshold_;
        }
    }
    return (infected / k1_) + (ill / k2_);
}

int Hodge::get_next_value_infected(int x, int y) {
    int check_x = 0, check_y = 0, offset = 0;
    int ill = 0, infected = 0, sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width_) % width_;
            check_y = (j + height_) % height_;
            offset = check_y * width_ + check_x;

            ill += board_[offset] == death_threshold_;
            infected += board_[offset] > 0 &&
                        board_[offset] < death_threshold_;
            if(board_[offset] > 0) {
                sum += board_[offset];
            }
        }
    }
    return sum / (ill + infected + 1) + infection_rate_;
}

int Hodge::get_sum_neighbors(int x, int y) {
    int check_x = 0, check_y = 0, offset = 0;
    int sum = 0;
    for(int i = x - 1; i <= x + 1; i++) {
        for(int j = y - 1; j <= y + 1; j++) {
            check_x = (i + width_) % width_;
            check_y = (j + height_) % height_;
            offset = check_y * width_ + check_x;

            if(board_[offset] > 0) {
                sum += board_[offset];
            }
        }
    }
    return sum;
}

void Hodge::print_rules() {
    std::cout << "Hodge";
    if(podge_) {
        std::cout << "podge: ";
        std::cout << "K1=" << k1_ << " ";
        std::cout << "K2=" << k2_ << " ";
    }
    else {
        std::cout << ": ";
        std::cout << "Infection Threshold=" << infection_threshold_ << " ";
    }
    std::cout << "Infection Rate=" << infection_rate_ << " ";
    std::cout << "Death threshold=" << death_threshold_ << " " << std::endl;
}

void Hodge::randomize_ruleset() {
    death_threshold_ = rand() % 400;
    k1_ = rand() % 5 + 1;
    k2_ = rand() % 5 + 1;
    infection_rate_ = rand() % 80;
    infection_threshold_ = rand() % 4 + 1;

    rainbows_.randomize_colors();
#ifdef USE_GPU
    if(use_gpu_) {
        copy_rules_to_gpu();
    }
#endif //USE_GPU
}

void Hodge::setup_cuda() {
#ifdef USE_GPU
    std::cout << "Starting CUDA" << std::endl;
    cudaMalloc((void**)&cudev_board_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_born_, 9 * sizeof(bool));
    cudaMalloc((void**)&cudev_stay_alive_, 9 * sizeof(bool));

    copy_board_to_gpu();
    copy_rules_to_gpu();
#endif //USE_GPU
}

void Hodge::tick() {
    if(use_gpu_) {
#if USE_GPU
        int *temp = NULL;

        call_cuda_lifelike(cudev_board_, cudev_board_buffer_, cudev_born_,
                           cudev_stay_alive_, num_faders_, width_, height_);

        cudaMemcpy(board_, cudev_board_buffer_, 
                   width_ * height_ * sizeof(int), cudaMemcpyDeviceToHost);

        temp = cudev_board_buffer_;
        cudev_board_buffer_ = cudev_board_;
        cudev_board_ = temp;
#endif //USE_GPU
    }
    else {
        update_board();
    }
}

void Hodge::update_board() {
    if(podge_) {
        update_hodgepodge();
    }
    else {
        update_hodge();
    }

    {
        int *tmp = board_buffer_;
        board_buffer_ = board_;
        board_ = tmp;
    }
}

void Hodge::update_hodge() {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            int offset = j * width_ + i;

            if(board_[offset] <= 0) {
                board_buffer_[offset] = (int)(get_sum_neighbors(i, j) >= infection_threshold_);
                if(changing_background_ && board_buffer_[offset] == 0) {
                    board_buffer_[offset] = board_[offset] - 1;
                }
            }
            else if(board_[offset] < death_threshold_) {
                board_buffer_[offset] = get_sum_neighbors(i, j) / 9;
                board_buffer_[offset] += infection_rate_;
            }
            else if(board_[offset] >= death_threshold_) {
                board_buffer_[offset] = 0;
            }
        }
    }
}

void Hodge::update_hodgepodge() {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            int offset = j * width_ + i;
            if(board_[offset] <= 0) {
                board_buffer_[offset] = get_next_value_healthy(i, j);
                if(changing_background_ && board_buffer_[offset] == 0) {
                    board_buffer_[offset] = board_[offset] - 1;
                }
            }
            else if(board_[offset] < death_threshold_) {
                board_buffer_[offset] = get_next_value_infected(i, j);
            }
            else if(board_[offset] >= death_threshold_) {
                board_buffer_[offset] = 0;
            }
        }
    }
}

