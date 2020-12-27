
#ifndef _LIFELIKE_H
#define _LIFELIKE_H

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class LifeLike : public Ruleset {
    private:
        int *board_;
        int *board_buffer_;
        bool born_[9];
        int current_tick_;
        Initializer initializer_;
        int num_faders_;
        Rainbows rainbows_;
        bool stay_alive_[9];

        void randomize_ruleset(bool control, bool shift);
        void update_board();
#ifdef USE_GPU
        int *cudev_board_;
        int *cudev_board_buffer_;
        bool *cudev_born_;
        bool *cudev_stay_alive_;

        void copy_board_to_gpu();
        void copy_rules_to_gpu();

        void start_cuda();
        void stop_cuda();
#endif

    public:
        LifeLike(int width, int height);
        ~LifeLike();
        void get_pixels(uint32_t *pixels);
        void print_controls();
        void print_rules();
        void start();
        void stop();
        void tick();
};

#endif //_LIFELIKE_H
