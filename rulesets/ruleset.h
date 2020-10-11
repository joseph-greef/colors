#ifndef _RULESET_H
#define _RULESET_H

#include <SDL.h>
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

    public:
        Ruleset(int width, int height);
        virtual ~Ruleset();

        virtual void free_cuda() = 0;
        virtual void get_pixels(uint32_t *pixels) = 0;
        virtual void handle_input(SDL_Event event, bool control, bool shift) = 0;
        virtual void print_rules() = 0;
        virtual void setup_cuda() = 0;

        virtual void tick() = 0;

        int get_num_alive_neighbors(int *board, int x, int y, int radius,
                                    NeighborhoodType type);
        void toggle_gpu();
};

#endif //_RULE_SET_H
