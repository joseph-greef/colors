#ifndef _RULESET_H
#define _RULESET_H

#include "SDL2/SDL.h"
#include <stdint.h>
#include <string>
#include <vector>

#include "board.cuh"

enum NeighborhoodType {
    VonNeuman,
    Moore,
};

namespace BoardType {
    enum BoardType {
        AgeBoard,
        PixelBoard,
        Other,
    };
};

class Ruleset {
    protected:
        bool use_gpu_;

        virtual void start_cuda() = 0;
        virtual void stop_cuda() = 0;

    public:
        Ruleset();
        virtual ~Ruleset();

        virtual BoardType::BoardType board_get_type() = 0;
        virtual BoardType::BoardType board_set_type() = 0;
        virtual void* get_board() = 0;
        virtual std::string get_name() = 0;
        virtual void get_pixels(Board<Pixel<uint8_t>> *pixels) = 0;
        virtual std::string get_rule_string() = 0;
        virtual void load_rule_string(std::string rules) = 0;
        virtual void print_human_readable_rules() = 0;
        virtual void set_board(void *new_board) = 0;
        virtual void start();
        virtual void stop();

        virtual void tick() = 0;

        void toggle_gpu();
};

#include "lifelike/lifelike.h"
#include "hodge/hodge.h"
#include "ants/ants.h"
#define NUM_RULESETS 3

#endif //_RULESET_H

