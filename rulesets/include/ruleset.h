#ifndef _RULESET_H
#define _RULESET_H

#include "SDL2/SDL.h"
#include <stdint.h>
#include <string>
#include <vector>

enum NeighborhoodType {
    VonNeuman,
    Moore,
};

class Ruleset {
    protected:
        int height_;
        bool use_gpu_;
        int width_;

#ifdef USE_GPU
        virtual void start_cuda() = 0;
        virtual void stop_cuda() = 0;
#endif //USE_GPU

    public:
        Ruleset(int width, int height);
        virtual ~Ruleset();

        virtual std::string get_name() = 0;
        virtual void get_pixels(uint32_t *pixels) = 0;
        virtual void print_human_readable_rules() = 0;
        virtual void start();
        virtual void stop();

        virtual void tick() = 0;

        int get_num_alive_neighbors(int *board, int x, int y, int radius,
                                    NeighborhoodType type);
        void toggle_gpu();
};

#include "lifelike/lifelike.h"
#include "hodge/hodge.h"
#include "ants/ants.h"
#define NUM_RULESETS 3

#endif //_RULESET_H
