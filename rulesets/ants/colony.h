#ifndef _ANTS_COLONY_H
#define _ANTS_COLONY_H

#include <random>
#include <vector>

class Colony;

#include "ant.h"

class Colony {
    private:
        int height_;
        int width_;
        int x_;
        int y_;
        int colony_number_;
        int color_;
        float food_signal_strength_;
        float home_signal_strength_;
        int max_signal_steps_;

        std::random_device rd_;
        std::mt19937 e2_;
        std::uniform_real_distribution<> dist_;

        float *food_pheromones_;
        float *food_pheromones_buffer_;

        float *home_pheromones_;
        float *home_pheromones_buffer_;

    public:
        Colony(int width, int height, int x, int y, int colony_number, int color);
        ~Colony();
        void add_starting_ants(std::vector<Ant> *ants);
        void draw_self(uint32_t *pixels);
        void move_ant(Ant *ant);
};

#endif //_ANTS_COLONY_H

