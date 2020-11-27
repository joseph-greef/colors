
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdlib.h>

#include "colony.h"

Colony::Colony(int width, int height, int x, int y, int colony_number, int color) 
    : height_(height)
    , width_(width)
    , x_(x)
    , y_(y)
    , colony_number_(colony_number)
    , color_(color)
    , e2_(rd_())
    , dist_(0, 10)
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

    food_signal_strength_ = dist_(e2_) / 10;
    home_signal_strength_ = dist_(e2_) / 10;
    max_signal_steps_ = static_cast<int>(dist_(e2_) + 5) * 20;

    food_pheromones_ = new float[width*height]();
    food_pheromones_buffer_ = new float[width*height]();

    home_pheromones_ = new float[width*height]();
    home_pheromones_buffer_ = new float[width*height]();

    std::cout << "New colony at " << x_ << ", " << y_ << std::endl;
    std::cout << "Food Signal: " << food_signal_strength_ << std::endl;
    std::cout << "Home Signal: " << home_signal_strength_ << std::endl;
}

Colony::~Colony() {
    delete [] food_pheromones_;
    delete [] food_pheromones_buffer_;

    delete [] home_pheromones_;
    delete [] home_pheromones_buffer_;
}

void Colony::add_starting_ants(std::vector<Ant> *ants) {
    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            if(i == 0 && j == 0) {
                continue;
            }
            ants->push_back({x_ + i,
                             y_ + j,
                             false,
                             0,
                             colony_number_,
                             this});
        }
    }

}
void Colony::draw_self(uint32_t *pixels) {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            int offset = j * width_ + i;
            int home_pheromone_adjusted;
            //if(home_pheromones_[offset] != 0) {
                //std::cout << i << " " << j << " " << home_pheromones_[offset] << std::endl;
            //}
            float home_pheromone = std::clamp(home_pheromones_[offset], 0.0f,
                                                                        255.0f);
            home_pheromone_adjusted = static_cast<int>(home_pheromone);
            pixels[offset] = home_pheromone_adjusted;

        }
    }
    for(int j = -2; j <= 2; j++) {
        for(int i = -2; i <= 2; i++) {
            if(i == 0 || j == 0) {
                int offset = (y_ + j) * width_ + (x_ + i);
                pixels[offset] = color_;
            }
        }
    }
}

void Colony::move_ant(Ant *ant) {
    float move_value[3*3] = { 0 };
    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            if(j == 1 && i == 1) {
                move_value[j * 3 + i] = -std::numeric_limits<float>::max();
            }
            else {
                int pheromone_offset = (ant->y + j - 1) * width_ + (ant->x + i - 1);
                if(!ant->has_food) {
                    move_value[j * 3 + i] += food_pheromones_[pheromone_offset];
                    move_value[j * 3 + i] -= home_pheromones_[pheromone_offset];
                }
                else {
                    move_value[j * 3 + i] += home_pheromones_[pheromone_offset];
                }
            }
        }
    }

    float max_value = -std::numeric_limits<float>::max();
    std::vector<int> max_i, max_j;
    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            //std::cout << i << " " << j << " " << move_value[j * 3 + i] << " | ";
            //This set of ifs will reset the maximum locations if the found
            //value is much larger than the current value, but if they're
            //roughly equal it'll just add to the maximum locations
            if(move_value[j * 3 + i] - max_value > 0.01) {
                max_value = move_value[j * 3 + i];
                max_i.clear();
                max_j.clear();
                max_i.push_back(i);
                max_j.push_back(j);
            }
            else if(move_value[j * 3 + i] - max_value > -0.01) {
                max_i.push_back(i);
                max_j.push_back(j);
            }
        }
    }
    //std::cout << std::endl;

    int movement_index = rand() % max_i.size();


    ant->x = (ant->x + max_i[movement_index] - 1) % width_;
    ant->y = (ant->y + max_j[movement_index] - 1) % height_;

    int pheromone_offset = ant->y * width_ + ant->x;
    if(ant->steps_since_event < max_signal_steps_) {
        if(ant->has_food) {
            food_pheromones_[pheromone_offset] +=
                static_cast<float>(max_signal_steps_ - ant->steps_since_event) *
                food_signal_strength_;
        }
        else {
            home_pheromones_[pheromone_offset] +=
                static_cast<float>(max_signal_steps_ - ant->steps_since_event) *
                home_signal_strength_;
        }
    }

    ant->steps_since_event += 1;
}
