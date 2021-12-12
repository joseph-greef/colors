
#ifndef _EMPTY_RULESET_H
#define _EMPTY_RULESET_H

#include "ruleset.h"


class EmptyRuleset : public Ruleset {
    private:
        void start_cuda();
        void stop_cuda();

    public:
        EmptyRuleset(int width, int height);
        ~EmptyRuleset();

        BoardType::BoardType board_get_type();
        BoardType::BoardType board_set_type();
        void* get_board();
        std::string get_name();
        void get_pixels(Board<Pixel<uint8_t>> *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void set_board(void *new_board);
        void start();
        void stop();
        void tick();
};

#endif //_EMPTY_RULESET_H
