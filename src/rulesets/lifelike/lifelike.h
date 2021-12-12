
#ifndef _LIFELIKE_H
#define _LIFELIKE_H

#include "board.cuh"

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class LifeLike : public Ruleset {
    private:
        Board<int> *board_;
        Board<int> *board_buffer_;

        bool born_[9];
        int current_tick_;
        Initializer initializer_;
        int num_faders_;
        Rainbows rainbows_;
        int random_fader_modulo_;
        bool stay_alive_[9];

        void randomize_ruleset();

        bool *cudev_born_;
        bool *cudev_stay_alive_;

        void copy_board_to_gpu();
        void copy_rules_to_gpu();

        void start_cuda();
        void stop_cuda();

    public:
        LifeLike(int width, int height);
        ~LifeLike();

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

#endif //_LIFELIKE_H
