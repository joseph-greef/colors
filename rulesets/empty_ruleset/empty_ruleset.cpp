
#include <iostream>

#ifdef USE_GPU
#include "cuda_runtime.h"
#include "empty_ruleset.cuh"
#endif

#include "input_manager.h"
#include "empty_ruleset.h"


EmptyRuleset::EmptyRuleset(int width, int height)
    : Ruleset(width, height)
{
#ifdef USE_GPU
    std::cout << "Allocating CUDA memory for EmptyRuleset" << std::endl;
#endif //USE_GPU

}

EmptyRuleset::~EmptyRuleset() {
#ifdef USE_GPU
    std::cout << "Freeing CUDA memory for EmptyRuleset" << std::endl;
#endif //USE_GPU
}

BoardType::BoardType EmptyRuleset::board_get_type() {
    return BoardType::Other;
}

BoardType::BoardType EmptyRuleset::board_set_type() {
    return BoardType::Other;
}

void* EmptyRuleset::get_board() {
    return NULL;
}

std::string EmptyRuleset::get_name() {
    return "EmptyRuleset";
}
void EmptyRuleset::get_pixels(uint32_t *pixels) {
}

std::string EmptyRuleset::get_rule_string() {
    return "";
}

void EmptyRuleset::load_rule_string(std::string rules) {
}

void EmptyRuleset::print_human_readable_rules() {
}

void EmptyRuleset::set_board(void* new_board) {
}

#ifdef USE_GPU
void EmptyRuleset::start_cuda() {
}

void EmptyRuleset::stop_cuda() {
}
#endif //USE_GPU

void EmptyRuleset::start() { 
    std::cout << "Starting EmptyRuleset" << std::endl;
    Ruleset::start();
}

void EmptyRuleset::stop() { 
    Ruleset::stop();
}

void EmptyRuleset::tick() {
    if(use_gpu_) {
#if USE_GPU
        call_cuda_empty_ruleset(width_, height_);
#endif //USE_GPU
    }
    else {
    }
}

