
#include <iostream>

#include "ants.h"

Ants::Ants(int width, int height)
    : Ruleset(width, height)
{
}

Ants::~Ants(){
}

void Ants::get_pixels(uint32_t *pixels){
    memset(pixels, 0, width_ * height_ * sizeof(uint32_t));
}

void Ants::handle_input(SDL_Event event, bool control, bool shift){
}

void Ants::print_controls(){
}

void Ants::print_rules(){
}

#ifdef USE_GPU
void Ants::start_cuda(){
}

void Ants::stop_cuda(){
}

#endif

void Ants::start(){
    std::cout << "Starting Ants" << std::endl;
}

void Ants::stop(){
}

void Ants::tick(){
}

