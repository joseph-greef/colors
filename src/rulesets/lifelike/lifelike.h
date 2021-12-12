
#ifndef _LIFELIKE_H
#define _LIFELIKE_H

#include "buffer.cuh"

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class LifeLike : public Ruleset {
    private:
        Buffer<int> *board_;
        Buffer<int> *board_buffer_;

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

        void copy_rules_to_gpu();

        void start_cuda();
        void stop_cuda();

    public:
        LifeLike(int width, int height);
        ~LifeLike();

        std::set<std::size_t> buffer_types_provided();
        std::size_t select_buffer_type(std::set<std::size_t> types);
        void* get_buffer(std::size_t type);
        void set_buffer(void *new_buffer, std::size_t type);

        std::string get_name();
        void get_pixels(Buffer<Pixel<uint8_t>> *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void start();
        void stop();
        void tick();
};

#endif //_LIFELIKE_H
