
#ifndef _HODGE_H
#define _HODGE_H

#include "initializer.h"
#include "rainbows.h"
#include "ruleset.h"


class Hodge : public Ruleset {
    private:
        Buffer<int> *board_;
        Buffer<int> *board_buffer_;
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
        void toggle_gpu();
};

#endif //_HODGE_H
