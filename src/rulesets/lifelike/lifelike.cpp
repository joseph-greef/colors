
#include <climits>
#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "lifelike.cuh"

#include "input_manager.h"
#include "lifelike.h"

LifeLike::LifeLike(int width, int height)
    : Ruleset()
    , initializer_(&board_, 1, 54)
    , num_faders_(0)
    , rainbows_(1)
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

    board_ = new Board<int>(width, height);
    board_buffer_ = new Board<int>(width, height);

    std::cout << "Allocating CUDA memory for LifeLike" << std::endl;
    cudaMalloc((void**)&cudev_born_, 9 * sizeof(int));
    cudaMalloc((void**)&cudev_stay_alive_, 9 * sizeof(bool));


    initializer_.init_center_cross();
}

LifeLike::~LifeLike() {
    delete board_;
    delete board_buffer_;

    std::cout << "Freeing CUDA memory for LifeLike" << std::endl;
    cudaFree((void*)cudev_born_);
    cudaFree((void*)cudev_stay_alive_);

}

/*
 * Cuda functions
 */
void LifeLike::copy_rules_to_gpu() {
    cudaMemcpy(cudev_born_, born_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaMemcpy(cudev_stay_alive_, stay_alive_, 9 * sizeof(bool),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void LifeLike::start_cuda() {
    copy_rules_to_gpu();
    board_->copy_host_to_device();
}

void LifeLike::stop_cuda() {
    board_->copy_device_to_host();
}

/*
 * Board Copy Functions:
 */
std::set<std::size_t> LifeLike::board_types_provided() {
    std::set<std::size_t> boards = { INT_BOARD };
    return boards;
}

std::size_t LifeLike::select_board_type(std::set<std::size_t> types) {
    if(types.find(INT_BOARD) != types.end()) {
        return INT_BOARD;
    }
    else {
        return NOT_COMPATIBLE;
    }
}

void* LifeLike::get_board(std::size_t type) {
    if(type == INT_BOARD) {
        return static_cast<void*>(board_);
    }
    else {
        return NULL;
    }
}

void LifeLike::set_board(void *new_board, std::size_t type) {
    if(type == INT_BOARD) {
        Board<int> *temp_board = static_cast<Board<int>*>(new_board);
        board_->copy_from_board(temp_board, use_gpu_);
    }
}

/*
 * Other Standard Ruleset Functions
 */
std::string LifeLike::get_name() {
    return "LifeLike";
}

void LifeLike::get_pixels(Board<Pixel<uint8_t>> *pixels) {
    rainbows_.age_to_pixels(board_, pixels, use_gpu_);
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

    if(use_gpu_) {
        copy_rules_to_gpu();
    }
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


    if(use_gpu_) {
        copy_rules_to_gpu();
    }

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
    /*
    if(initializer_.was_board_changed()) {
        current_tick_ = num_faders_;

        if(use_gpu_) {
            copy_board_to_gpu();
        }
    }
    */

    if(use_gpu_) {
        call_lifelike_kernel(board_, board_buffer_, cudev_born_,
                             cudev_stay_alive_, num_faders_, current_tick_);

        cudaDeviceSynchronize();
    }
    else {
        for(int i = 0; i < board_->width_ * board_->height_; i++) {
            lifelike_step(board_, board_buffer_, i, born_, stay_alive_,
                          num_faders_, current_tick_);
        }
    }
    Board<int> *tmp = board_buffer_;
    board_buffer_ = board_;
    board_ = tmp;

    current_tick_++;
}

/*
 * LifeLike Specific Functions
 */

