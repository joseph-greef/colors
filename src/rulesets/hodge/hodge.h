
#ifndef _HODGE_H
#define _HODGE_H

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class Hodge : public Ruleset {
    private:
        Board<int> *board_;
        Board<int> *board_buffer_;
        int death_threshold_;
        int infection_rate_;
        int infection_threshold_;
        Initializer initializer_;
        int k1_;
        int k2_;
        bool podge_;
        Rainbows rainbows_;

        void start_cuda();
        void stop_cuda();

        void randomize_ruleset();

    public:
        Hodge(int width, int height);
        ~Hodge();

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
        void toggle_gpu();
};

#endif //_HODGE_H
