
#ifndef _EMPTY_RULESET_H
#define _EMPTY_RULESET_H

#include "ruleset.h"


class EmptyRuleset : public Ruleset {
    private:
#ifdef USE_GPU
        void start_cuda();
        void stop_cuda();
#endif

    public:
        EmptyRuleset(int width, int height);
        ~EmptyRuleset();

        std::string get_name();
        void get_pixels(uint32_t *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void start();
        void stop();
        void tick();
};

#endif //_EMPTY_RULESET_H
