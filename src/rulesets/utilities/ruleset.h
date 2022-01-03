#ifndef _RULESET_H
#define _RULESET_H

#include "SDL2/SDL.h"
#include <set>
#include <stdint.h>
#include <string>

#include "buffer.cuh"


class Ruleset {
    protected:
        bool use_gpu_;

        virtual void start_cuda() = 0;
        virtual void stop_cuda() = 0;

    public:
        Ruleset();
        virtual ~Ruleset();

        virtual std::set<std::size_t> buffer_types_provided() = 0;
        virtual std::size_t select_buffer_type(std::set<std::size_t> types) = 0;
        virtual void* get_buffer(std::size_t type) = 0;
        virtual void set_buffer(void *new_buffer, std::size_t type) = 0;

        virtual std::string get_name() = 0;
        virtual void get_pixels(Buffer<Pixel<uint8_t>> *pixels) = 0;
        virtual std::string get_rule_string() = 0;
        virtual void load_rule_string(std::string rules) = 0;
        virtual void print_human_readable_rules() = 0;
        virtual void start();
        virtual void stop();

        virtual void tick() = 0;

        void toggle_gpu();
};

#include "lifelike/lifelike.h"
#include "hodge/hodge.h"
#include "ants/ants.h"
#include "empty_ruleset/empty_ruleset.h"
#define NUM_RULESETS 4

#endif //_RULESET_H

