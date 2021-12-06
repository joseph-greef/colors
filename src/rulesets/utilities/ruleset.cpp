
#include <climits>
#include <cmath>
#include <iostream>

#include "input_manager.h"
#include "ruleset.h"
#include "rainbows.h"

Ruleset::Ruleset()
    : use_gpu_(false)
{
}

Ruleset::~Ruleset() {
}

void Ruleset::start() {
    ADD_FUNCTION_CALLER(&Ruleset::toggle_gpu, SDL_SCANCODE_F, false, false,
                        "Game", "Toggle CUDA processing");
}

void Ruleset::stop() {
    InputManager::remove_var_changer(SDL_SCANCODE_F, false, false);
}

void Ruleset::toggle_gpu() {
    use_gpu_ = !use_gpu_;

    if(use_gpu_) {
        start_cuda();
        std::cout << "Starting CUDA" << std::endl;
    }
    else {
        stop_cuda();
        std::cout << "Stopping CUDA" << std::endl;
    }

}

