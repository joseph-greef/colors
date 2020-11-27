#ifndef _ANTS_COLONY_H
#define _ANTS_COLONY_H

#include <vector>

#include "ant_location.h"

class Colony {
    private:
        int height_;
        int width_;
        int x_;
        int y_;
        int colony_number_;
        int color_;

        std::vector<AntLocation> ant_locations_;

        float *enemy_pheromones_;
        float *enemy_pheromones_buffer_;


    public:
        Colony(int width, int height, int x, int y, int colony_number, int color);
        ~Colony();
        void draw_self(uint32_t *pixels);
};

#endif //_ANTS_COLONY_H

