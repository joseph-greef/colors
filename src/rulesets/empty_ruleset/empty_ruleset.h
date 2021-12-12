
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

        std::set<std::size_t> board_types_provided();
        std::size_t select_board_type(std::set<std::size_t> types);
        void* get_board(std::size_t type);
        void set_board(void *new_board, std::size_t type);

        std::string get_name();
        void get_pixels(Board<Pixel<uint8_t>> *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void start();
        void stop();
        void tick();
};

#endif //_EMPTY_RULESET_H
