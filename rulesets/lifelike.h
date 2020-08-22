
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
        int dot_radius_;
        int density_;
        Initializer initializer_;
        int num_faders_;
        Rainbows rainbows_;
        bool stay_alive_[9];

#ifdef USE_GPU
        int *cudev_board_;
        int *cudev_board_buffer_;
        bool *cudev_born_;
        bool *cudev_stay_alive_;
#endif

        void copy_board_to_gpu();
        void copy_rules_to_gpu();

        void randomize_ruleset();
        void update_board();


    public:
        LifeLike(int width, int height);
        ~LifeLike();
        void free_cuda();
        void get_pixels(uint32_t *pixels);
        void handle_input(SDL_Event event, bool control, bool shift);
        void print_rules();
        void setup_cuda();
        void tick();
};

#endif //_LIFELIKE_H
