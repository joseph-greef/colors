
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdlib.h>

#include "opencv2/opencv.hpp"
#include "colony.h"

Colony::Colony(int width, int height, int x, int y, uint32_t color) 
    : height_(height)
    , width_(width)
    , x_(x)
    , y_(y)
    , color_(color)
    , e2_(rd_())
    , dist_(0, 1)
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
        y_ = height_ - 3;
    }

    aggression_ = dist_(e2_);
    enemy_encounter_amount_ = dist_(e2_) * 500 + 500;
    enemy_signal_strength_ = dist_(e2_) * 10 + 5;
    food_signal_strength_ = dist_(e2_) * 10 + 5;
    home_signal_strength_ = dist_(e2_) * 70 + 35;
    home_smell_amount_ = dist_(e2_) * 20 + 15;
    randomness_ = dist_(e2_) * 2;
    max_signal_steps_ = static_cast<int>(dist_(e2_) * 200 + 100);
    max_total_steps_ = static_cast<int>(dist_(e2_) * 500 + 1000);

    food_pheromones_ = new float[width*height]();
    food_pheromones_buffer_ = new float[width*height]();

    home_pheromones_ = new float[width*height]();
    home_pheromones_buffer_ = new float[width*height]();

    enemy_pheromones_ = new float[width*height]();
    enemy_pheromones_buffer_ = new float[width*height]();

    std::cout << "New colony at " << x_ << ", " << y_ << std::endl;
    std::cout << "Food Signal: " << food_signal_strength_ << std::endl;
    std::cout << "Home Signal: " << home_signal_strength_ << std::endl;
}

Colony::~Colony() {
    for(Ant *ant: ants_) {
        delete ant;
    }
    delete [] food_pheromones_;
    delete [] food_pheromones_buffer_;

    delete [] home_pheromones_;
    delete [] home_pheromones_buffer_;

    delete [] enemy_pheromones_;
    delete [] enemy_pheromones_buffer_;
}

void Colony::add_ants(std::list<Ant*> *ants, int number) {
    for(int i = 0; i < number; i++) {
        ants->push_back(new Ant(x_ + rand() % 3 - 1,
                                y_ + rand() % 3 - 1,
                                this));
        ants_.push_back(ants->back());
    }
}

void Colony::add_enemy_smell(int x, int y, float amount) {
    enemy_pheromones_[y * width_ + x] += amount;
}

void Colony::add_food_smell(int x, int y, float amount) {
    food_pheromones_[y * width_ + x] += amount;
}

void Colony::draw_pheromones(uint32_t *pixels) {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            int offset = j * width_ + i;
            //if(home_pheromones_[offset] != 0) {
                //std::cout << i << " " << j << " " << home_pheromones_[offset] << std::endl;
            //}
            float home_pheromone = std::clamp(home_pheromones_[offset], 0.0f,
                                                                        255.0f);
            int home_pheromone_adjusted = static_cast<int>(home_pheromone);

            float food_pheromone = std::clamp(food_pheromones_[offset], 0.0f,
                                                                        255.0f);
            int food_pheromone_adjusted = static_cast<int>(food_pheromone);

            float enemy_pheromone = std::clamp(enemy_pheromones_[offset], 0.0f,
                                                                        255.0f);
            int enemy_pheromone_adjusted = static_cast<int>(enemy_pheromone);

            pixels[offset] = (home_pheromone_adjusted << 0) +
                             (food_pheromone_adjusted << 8) +
                             (enemy_pheromone_adjusted << 16);
        }
    }
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

//return whether this colony's ant won
bool Colony::enemy_encountered(Ant *ant, Ant *enemy_ant,
                               float roll, float enemy_roll) {
    enemy_pheromones_[ant->y * width_ + ant->x] += enemy_encounter_amount_;

    if(enemy_ant->colony->get_aggression() + enemy_roll >= aggression_ + roll) {
        ants_.remove(ant);
        return false;
    }
    return true;
}

std::list<Ant*>* Colony::get_ants() {
    return &ants_;
}

float Colony::get_aggression() {
    return aggression_;
}

uint32_t Colony::get_color() {
    return color_;
}

int Colony::get_offset() {
    return y_ * width_ + x_;
}

uint32_t Colony::get_num_ants() {
    return ants_.size();
}

int Colony::get_x() {
    return x_;
}

int Colony::get_y() {
    return y_;
}

bool Colony::move_ant(Ant *ant) {
    ant->steps_since_event += 1;
    ant->total_steps += 1;
    if(ant->total_steps > max_total_steps_) {
        ants_.remove(ant);
        delete ant;
        return false;
    }

    float move_value[3*3] = { 0 };
    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            if(j == 1 && i == 1) {
                move_value[j * 3 + i] = -std::numeric_limits<float>::max();
            }
            else {
                int tmp_x = ((ant->x + i - 1) + width_) % width_;
                int tmp_y = ((ant->y + j - 1) + height_) % height_;
                int pheromone_offset = tmp_y * width_ + tmp_x;
                if(!ant->has_food) {
                    move_value[j * 3 + i] += food_pheromones_[pheromone_offset];
                    move_value[j * 3 + i] += aggression_ *
                                             enemy_pheromones_[pheromone_offset];
                    move_value[j * 3 + i] -= home_pheromones_[pheromone_offset]/10;
                    //enemy_pheromones_[pheromone_offset] *= 0.9;
                    food_pheromones_[pheromone_offset] *= 0.9;
                }
                else {
                    move_value[j * 3 + i] += home_pheromones_[pheromone_offset];
                }
                //std::cout << move_value[j * 3 + i] << " ";
                if(j != 1 && i != 1) { //corner squares get sqrt(2) value decrease
                    if(move_value[j * 3 + i] > 0) {
                        move_value[j * 3 + i] /= 1.075;
                    }
                    else {
                        move_value[j * 3 + i] *= 1.075;
                    }
                }    
            }
        }
    }

    float max_value = -std::numeric_limits<float>::max();
    for(int i = 0; i < 3 * 3; i++) {
        //std::cout << move_value[j * 3 + i] << " ";
        //This set of ifs will reset the maximum locations if the found
        //value is much larger than the current value, but if they're
        //roughly equal it'll just add to the maximum locations
        if(move_value[i] > max_value) {
            max_value = move_value[i];
        }
    }

    std::vector<int> max_i, max_j;
    for(int j = 0; j < 3; j++) {
        for(int i = 0; i < 3; i++) {
            if(move_value[j * 3 + i] > max_value - randomness_) {
                max_i.push_back(i);
                max_j.push_back(j);
            }
        }
    }

    int movement_index = rand() % max_i.size();

    ant->x = (ant->x + max_i[movement_index] - 1);
    ant->y = (ant->y + max_j[movement_index] - 1);

    if(ant->x < 1) {
        ant->x = 1;
    }
    else if(ant->x > width_ - 2) {
        ant->x = width_ - 2;
    }
    if(ant->y < 1) {
        ant->y = 1;
    }
    else if(ant->y > height_ - 2) {
        ant->y = height_ - 2;
    }


    int pheromone_offset = ant->y * width_ + ant->x;
    if(ant->enemy_seen) {
        enemy_pheromones_[pheromone_offset] +=
            (max_signal_steps_/static_cast<float>(ant->steps_since_event)) *
            enemy_signal_strength_;
    }
    else if(ant->has_food) {
        food_pheromones_[pheromone_offset] +=
            (max_signal_steps_/static_cast<float>(ant->steps_since_event)) *
            food_signal_strength_;
    }
    else {
        home_pheromones_[pheromone_offset] +=
            (max_signal_steps_/static_cast<float>(ant->steps_since_event)) *
            home_signal_strength_;
    }
    return true;
}

bool Colony::owns_ant(Ant *ant) {
    return ant->colony == this;
}

void Colony::update_pheromones() {
    float pheromone_decay = 0.999;
    home_pheromones_[y_ * width_ + x_] += home_smell_amount_;

    home_pheromones_[y_ * width_ + x_] += max_signal_steps_ * home_signal_strength_;
    cv::Mat home(height_, width_, CV_32F, home_pheromones_);
    cv::Mat home_buffer(height_, width_, CV_32F, home_pheromones_buffer_);
    cv::GaussianBlur(home, home_buffer, cv::Size(17, 17), 0);
    home_buffer *= pheromone_decay;

    cv::Mat food(height_, width_, CV_32F, food_pheromones_);
    cv::Mat food_buffer(height_, width_, CV_32F, food_pheromones_buffer_);
    cv::GaussianBlur(food, food_buffer, cv::Size(5, 5), 0);
    home_buffer *= pheromone_decay;

    cv::Mat enemy(height_, width_, CV_32F, enemy_pheromones_);
    cv::Mat enemy_buffer(height_, width_, CV_32F, enemy_pheromones_buffer_);
    cv::GaussianBlur(enemy, enemy_buffer, cv::Size(11, 11), 0);
    home_buffer *= pheromone_decay;

    float *tmp = home_pheromones_;
    home_pheromones_ = home_pheromones_buffer_;
    home_pheromones_buffer_ = tmp;

    tmp = food_pheromones_;
    food_pheromones_ = food_pheromones_buffer_;
    food_pheromones_buffer_ = tmp;

    tmp = enemy_pheromones_;
    enemy_pheromones_ = enemy_pheromones_buffer_;
    enemy_pheromones_buffer_ = tmp;
}
