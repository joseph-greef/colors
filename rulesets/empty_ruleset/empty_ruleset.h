
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
        void get_pixels(uint32_t *pixels);
        void print_rules();
        void start();
        void stop();
        void tick();
};

#endif //_EMPTY_RULESET_H
