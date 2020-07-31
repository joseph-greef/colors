#ifndef _RULESET_H
#define _RULESET_H

#include <stdint.h>

enum NeighborhoodType {
    VonNeuman,
    Moore,
};

class Ruleset {
    protected:
        int _height;
        int _width;

    public:
        Ruleset(int width, int height);
        virtual void tick() = 0;
        virtual void get_pixels(uint32_t *pixels) = 0;

        int get_num_alive_neighbors(int *board, int x, int y, int radius,
                                    NeighborhoodType type);
};

#endif //_RULE_SET_H
