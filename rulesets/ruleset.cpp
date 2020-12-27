
#include <climits>
#include <cmath>
#include <iostream>

#include "input_manager.h"
#include "ruleset.h"
#include "rainbows.h"

Ruleset::Ruleset(int width, int height)
    : height_(height)
    , use_gpu_(false)
    , width_(width)
{
}

Ruleset::~Ruleset() {
}

int Ruleset::get_num_alive_neighbors(int *board, int x, int y,
                                     int radius,
                                     NeighborhoodType type) {
    int check_x = 0;
    int check_y = 0; 
    int count = 0;

    if(type == VonNeuman) {
        for(int i = x - radius; i <= x + radius; i++) {
            for(int j = y - radius; j <= y + radius; j++) {
                if(j==y && i==x)
                    continue;
                if(abs(i-x)+abs(j-y) <= radius) {
                    check_x = (i + width_) % width_;
                    check_y = (j + height_) % height_;
                    //and check the coordinate, if it's alive increase count
                    if(board[check_y*width_+check_x] > 0)
                        count++;
                }
            }
        }
    }
    else {
        for(int i = x - radius; i <= x + radius; i++) {
            for(int j = y - radius; j <= y + radius; j++) {
                if(j==y && i==x)
                    continue;


                check_x = (i + width_) % width_;
                check_y = (j + height_) % height_;
                //and check the coordinate, if it's alive increase count
                if(board[check_y*width_+check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}

void Ruleset::start() {
    ADD_FUNCTION_CALLER(&Ruleset::toggle_gpu, SDLK_f,
                        "(Game) Toggle CUDA processing");
}

void Ruleset::stop() {
    InputManager::remove_var_changer(SDLK_f);
}

void Ruleset::toggle_gpu(bool control, bool shift) {
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

