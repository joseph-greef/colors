#include <climits>
#include <iostream>
#include <stdlib.h>

#include "curand.h"
#include "cuda_runtime.h"
#include "hodge.cuh"

#include "input_manager.h"
#include "hodge.h"

Hodge::Hodge(int width, int height)
    : Ruleset()
    , death_threshold_(260)
    , infection_rate_(30)
    , infection_threshold_(2)
    , initializer_(&board_, 2, 5)
    , k1_(2)
    , k2_(5)
    , podge_(true)
    , rainbows_(1)
{
    board_ = new Board<int>(width, height);
    board_buffer_ = new Board<int>(width, height);

    initializer_.init_center_square();
}

Hodge::~Hodge() {
    delete board_;
    delete board_buffer_;
}


void Hodge::start_cuda() {
    board_->copy_host_to_device();
}

void Hodge::stop_cuda() {
    board_->copy_device_to_host();
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

void Hodge::get_pixels(Board<Pixel<uint8_t>> *pixels) {
    rainbows_.age_to_pixels(board_, pixels, use_gpu_);
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
    //memcpy(board_, new_board, width_ * height_* sizeof(board_[0]));
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
        if(podge_) {
            call_hodgepodge_kernel(board_, board_buffer_,
                                   death_threshold_,
                                   infection_rate_, k1_, k2_);
        }
        else {
            call_hodge_kernel(board_, board_buffer_, death_threshold_,
                              infection_rate_, infection_threshold_);
        }

        cudaDeviceSynchronize();
    }
    else {
        for(int i = 0; i < board_->width_ * board_->height_; i++) {
            if(podge_) {
                hodgepodge_step(board_, board_buffer_, i, death_threshold_,
                                infection_rate_, k1_, k2_);
            }
            else {
                hodge_step(board_, board_buffer_, i, death_threshold_,
                           infection_rate_, infection_threshold_);
            }
        }
    }
    {
        Board<int> *tmp = board_buffer_;
        board_buffer_ = board_;
        board_ = tmp;
    }
}

