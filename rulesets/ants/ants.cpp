
#include <iostream>
#include <stdlib.h>

#include "ants.h"

Ants::Ants(int width, int height)
    : Ruleset(width, height)
    , num_colonies_(5)
{
    world_ = new int[width_ * height_];
    reset();
}

Ants::~Ants() {
    for(Colony *colony: colonies_) {
        delete colony;
    }
    delete world_;
}

void Ants::get_pixels(uint32_t *pixels){
    memset(pixels, 0, width_ * height_ * sizeof(uint32_t));
    for(Colony *colony: colonies_) {
        colony->draw_self(pixels);
    }
}

void Ants::handle_input(SDL_Event event, bool control, bool shift){
    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_e:
                reset();
                break;
        }
    }
}

void Ants::print_controls(){
}

void Ants::print_rules(){
}

void Ants::reset() {
    for(Colony *colony: colonies_) {
        delete colony;
    }
    colonies_.clear();
    for(int i = 0; i < num_colonies_; i++) {
        colonies_.push_back(new Colony(width_, height_,
                                       rand() % width_, rand() % height_,
                                       i+1, colony_colors[i+1]));
    }
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


uint32_t Ants::colony_colors[] = {
    0x000000, //Junk color at 0, because we'll never use it
    0xFFFFFF,
    0x0000FF,
    0xFFFF00,
    0xFF00FF,
    0xb06000,
};

int Ants::max_colonies = sizeof(colony_colors) / sizeof(colony_colors[0]);
