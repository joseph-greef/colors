#include <climits>
#include <iostream>
#include <stdlib.h>

#include "curand.h"
#include "cuda_runtime.h"
#include "hodge.cuh"

#include "input_manager.h"
#include "hodge.h"

Hodge::Hodge(int width, int height)
    : Ruleset(width, height)
    , death_threshold_(260)
    , infection_rate_(30)
    , infection_threshold_(2)
    , initializer_(&board_, 2, 5, width, height)
    , k1_(2)
    , k2_(5)
    , podge_(true)
    , rainbows_(width, height, 1)
{
    board_ = new int[width*height];
    board_buffer_ = new int[width*height];


    std::cout << "Allocating CUDA memory for Hodge" << std::endl;
    cudaMalloc((void**)&cudev_board_, width_ * height_ * sizeof(int));
    cudaMalloc((void**)&cudev_board_buffer_, width_ * height_ * sizeof(int));


    initializer_.init_center_square();
}

Hodge::~Hodge() {
    delete [] board_;
    delete [] board_buffer_;

    std::cout << "Freeing CUDA memory for Hodge" << std::endl;
    cudaFree((void*)cudev_board_);
    cudaFree((void*)cudev_board_buffer_);

}


void Hodge::copy_board_to_gpu() {
    cudaMemcpy(cudev_board_, board_, width_ * height_ * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void Hodge::start_cuda() {
    copy_board_to_gpu();
}

void Hodge::stop_cuda() {
}


BoardType::BoardType Hodge::board_get_type() {
    return BoardType::AgeBoard;
}

BoardType::BoardType Hodge::board_set_type() {
    return BoardType::AgeBoard;
}

void* Hodge::get_board() {
    return static_cast<void*>(board_);
}

std::string Hodge::get_name() {
    return "Hodge";
}

void Hodge::get_pixels(uint32_t *pixels) {
    rainbows_.age_to_pixels(board_, pixels);
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

std::string Hodge::get_rule_string() {
    std::ostringstream rule_ss;
    rule_ss << death_threshold_ << " ";
    rule_ss << infection_rate_ << " ";
    rule_ss << infection_threshold_ << " ";
    rule_ss << k1_ << " ";
    rule_ss << k2_ << " ";
    rule_ss << podge_;
    return rule_ss.str();
}

void Hodge::load_rule_string(std::string rules) {
    std::istringstream rule_ss(rules);
    rule_ss >> death_threshold_;
    rule_ss >> infection_rate_;
    rule_ss >> infection_threshold_;
    rule_ss >> k1_;
    rule_ss >> k2_;
    rule_ss >> podge_;
}

void Hodge::print_human_readable_rules() {
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
}

void Hodge::set_board(void *new_board) {
    memcpy(board_, new_board, width_ * height_* sizeof(board_[0]));
}

void Hodge::start() { 
    std::cout << "Starting Hodge" << std::endl;
    Ruleset::start();

    InputManager::add_bool_toggler(&podge_, SDL_SCANCODE_T, false, false,
                                   "Hodge", "Toggle between Hodgepodge and Hodge");

    ADD_FUNCTION_CALLER(&Hodge::randomize_ruleset, SDL_SCANCODE_R, false, false,
                        "Hodge", "Randomize Ruleset");

    InputManager::add_int_changer(&infection_rate_, SDL_SCANCODE_A,
                                  false, false, INT_MIN, INT_MAX,
                                  "Hodge", "Infection Rate");
    InputManager::add_int_changer(&death_threshold_, SDL_SCANCODE_D,
                                  false, false, 0, INT_MAX,
                                  "Hodge", "Death Threshold");
    InputManager::add_int_changer(&infection_threshold_, SDL_SCANCODE_S,
                                  false, false, 0, INT_MAX,
                                  "Hodge", "Infection Theshold");
    InputManager::add_int_changer(&k1_, SDL_SCANCODE_Q,
                                  false, false, 0, INT_MAX,
                                  "Hodge", "k1");
    InputManager::add_int_changer(&k2_, SDL_SCANCODE_W,
                                  false, false, 0, INT_MAX,
                                  "Hodge", "k2");

    initializer_.start();
    rainbows_.start();
}

void Hodge::stop() { 
    Ruleset::stop();

    InputManager::remove_var_changer(SDL_SCANCODE_T, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_R, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_A, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_D, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Q, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_S, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_W, false, false);

    initializer_.stop();
    rainbows_.stop();
}

void Hodge::tick() {
    if(use_gpu_) {

        int *temp = NULL;
        if(initializer_.was_board_changed()) {
            copy_board_to_gpu();
        }
        if(podge_) {
            call_cuda_hodgepodge(cudev_board_, cudev_board_buffer_, death_threshold_,
                                 infection_rate_, k1_, k2_,
                                 width_, height_);
        }
        else {
            call_cuda_hodge(cudev_board_, cudev_board_buffer_, death_threshold_,
                            infection_rate_, infection_threshold_,
                            width_, height_);
        }

        cudaMemcpy(board_, cudev_board_buffer_, 
                   width_ * height_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        temp = cudev_board_buffer_;
        cudev_board_buffer_ = cudev_board_;
        cudev_board_ = temp;

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

