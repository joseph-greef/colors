
#include <iostream>

#include "cuda_runtime.h"
#include "empty_ruleset.cuh"

#include "input_manager.h"
#include "empty_ruleset.h"


EmptyRuleset::EmptyRuleset(int width, int height)
    : Ruleset()
{
}

EmptyRuleset::~EmptyRuleset() {
}

/*
 * Cuda Functions:
 */
void EmptyRuleset::start_cuda() {
}

void EmptyRuleset::stop_cuda() {
}

/*
 * Board Copy Functions:
 */
std::set<std::size_t> EmptyRuleset::board_types_provided() {
    std::set<std::size_t> boards = { };
    return boards;
}

std::size_t EmptyRuleset::select_board_type(std::set<std::size_t> types) {
    return NOT_COMPATIBLE;
}

void* EmptyRuleset::get_board(std::size_t type) {
    return NULL;
}

void EmptyRuleset::set_board(void *new_board, std::size_t type) {
}

/*
 * Other Standard Ruleset Functions
 */
std::string EmptyRuleset::get_name() {
    return "EmptyRuleset";
}
void EmptyRuleset::get_pixels(Board<Pixel<uint8_t>> *pixels) {
}

std::string EmptyRuleset::get_rule_string() {
    return "";
}

void EmptyRuleset::load_rule_string(std::string rules) {
}

void EmptyRuleset::print_human_readable_rules() {
}

void EmptyRuleset::start() {
    std::cout << "Starting EmptyRuleset" << std::endl;
    Ruleset::start();
}

void EmptyRuleset::stop() {
    Ruleset::stop();
}

void EmptyRuleset::tick() {
    if(use_gpu_) {

        call_cuda_empty_ruleset();

    }
    else {
    }
}

/*
 * EmptyRuleset Specific Functions
 */
