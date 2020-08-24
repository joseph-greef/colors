
#ifndef _HODGE_H
#define _HODGE_H

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class Hodge : public Ruleset {
    private:
        int *board_;
        int *board_buffer_;
        bool changing_background_;
        int death_threshold_;
        int infection_rate_;
        int infection_threshold_;
        Initializer initializer_;
        int k1_;
        int k2_;
        bool podge_;
        Rainbows rainbows_;

#ifdef USE_GPU
        int *cudev_board_;
        int *cudev_board_buffer_;
        bool *cudev_born_;
        bool *cudev_stay_alive_;
#endif

        void copy_board_to_gpu();
        void copy_rules_to_gpu();

        int get_next_value_healthy(int x, int y);
        int get_next_value_infected(int x, int y);
        int get_sum_neighbors(int x, int y);

        void randomize_ruleset();
        void update_board();
        void update_hodge();
        void update_hodgepodge();


    public:
        Hodge(int width, int height);
        ~Hodge();

        static std::string Name;

        void free_cuda();
        void get_pixels(uint32_t *pixels);
        void handle_input(SDL_Event event, bool control, bool shift);
        void print_rules();
        void setup_cuda();
        void tick();
};

#endif //_HODGE_H