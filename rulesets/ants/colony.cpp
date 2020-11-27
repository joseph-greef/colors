
#include <iostream>

#include "colony.h"

Colony::Colony(int width, int height, int x, int y, int colony_number, int color) 
    : height_(height)
    , width_(width)
    , x_(x)
    , y_(y)
    , colony_number_(colony_number)
    , color_(color)
{
    if(x_ < 3) {
        x_ = 3;
    }
    else if(x_ > width_ - 3) {
        x_ = width_ - 3;
    }
    if(y_ < 3) {
        y_ = 3;
    }
    else if(y_ > height_ - 3) {
        y_ = height - 3;
    }

    std::cout << "New colony at " << x_ << ", " << y_ << std::endl;

    enemy_pheromones_ = new float[width*height];
    enemy_pheromones_buffer_ = new float[width*height];

    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            if(i == 0 && j == 0) {
                continue;
            }
            ant_locations_.push_back({x_ + i, y_ + j});
        }
    }
}

Colony::~Colony() {
    delete [] enemy_pheromones_;
    delete [] enemy_pheromones_buffer_;
}

void Colony::draw_self(uint32_t *pixels) {
    for(int j = -2; j <= 2; j++) {
        for(int i = -2; i <= 2; i++) {
            if(i == 0 || j == 0) {
                int offset = (y_ + j) * width_ + (x_ + i);
                pixels[offset] = color_;
            }
        }
    }
}
