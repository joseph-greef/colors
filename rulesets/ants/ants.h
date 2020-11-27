#ifndef _ANTS_ANTS_H
#define _ANTS_ANTS_H

#include <vector>

#include "../ruleset.h"
#include "ant_location.h"
#include "colony.h"

class Ants : public Ruleset {
    private:
        std::vector<Colony*> colonies_;
        int num_colonies_;
        int *world_;

        void reset();
#ifdef USE_GPU
        void start_cuda();
        void stop_cuda();
#endif

        static uint32_t colony_colors[];
        static int max_colonies;

    public:
        Ants(int width, int height);
        ~Ants();
        void get_pixels(uint32_t *pixels);
        void handle_input(SDL_Event event, bool control, bool shift);
        void print_controls();
        void print_rules();
        void start();
        void stop();
        void tick();
};

#endif //_ANTS_ANTS_H
