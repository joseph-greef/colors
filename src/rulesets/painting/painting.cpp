
#include <iostream>

#include "cuda_runtime.h"
#include "painting.cuh"

#include "input_manager.h"
#include "painting.h"


Painting::Painting(int width, int height)
    : Ruleset()
    , current_stroke_()
    , rainbows_(1)
    , time_(0)
{
    age_board_ = new Buffer<int>(width, height);
}

Painting::~Painting() {
    delete age_board_;
}

/*
 * CUDA Functions
 */
void Painting::start_cuda() {
    use_gpu_ = true;
}

void Painting::stop_cuda() {
    use_gpu_ = false;
}

/*
 * Buffer Copy Functions:
 */
std::set<std::size_t> Painting::buffer_types_provided() {
    std::set<std::size_t> buffers= { INT_BUFFER };
    return buffers;
}

std::size_t Painting::select_buffer_type(std::set<std::size_t> types) {
    return NOT_COMPATIBLE;
}

void* Painting::get_buffer(std::size_t type) {
    return NULL;
}

void Painting::set_buffer(void* buffer, std::size_t type) {
}

/*
 * Other Standard Ruleset Functions
 */
std::string Painting::get_name() {
    return "Painting";
}

void Painting::get_pixels(Buffer<Pixel<uint8_t>> *pixels) {
    rainbows_.age_to_pixels(age_board_, pixels, use_gpu_);
}

std::string Painting::get_rule_string() {
    return "";
}

void Painting::load_rule_string(std::string rules) {
}

void Painting::print_human_readable_rules() {
}

void Painting::start() {
    std::cout << "Starting Painting" << std::endl;
    Ruleset::start();
    rainbows_.start();
}

void Painting::stop() {
    Ruleset::stop();
    rainbows_.stop();
}

void Painting::tick() {
    //if(use_gpu_) {
    //    call_cuda_painting();
    //}
    //else {
        current_stroke_.mark(age_board_, time_);
    //}
    time_ += 1;
}

/*
 * Painting Specific Functions
 */

