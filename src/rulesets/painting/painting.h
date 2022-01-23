
#ifndef _PAINTING_H
#define _PAINTING_H

#include "buffer.cuh"
#include "rainbows.h"
#include "ruleset.h"

#include "stroke.h"


class Painting : public Ruleset {
    private:
        Rainbows rainbows_;

        void start_cuda();
        void stop_cuda();

    public:
        Painting(int width, int height);
        ~Painting();

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

#endif //_PAINTING_H
