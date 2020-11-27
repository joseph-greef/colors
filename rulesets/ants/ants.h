#ifndef _ANTS_ANTS_H
#define _ANTS_ANTS_H

#include "../ruleset.h"

class Ants : public Ruleset {
    private:
#ifdef USE_GPU
        void start_cuda();
        void stop_cuda();
#endif

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
