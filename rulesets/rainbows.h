#ifndef _RAINBOW_H
#define _RAINBOW_H

#include <stdint.h>

class Rainbows {
    public:
        static void age_to_pixels(int *age_board, uint32_t *pixels, 
                                  int alive_color_scheme, int alive_offset,
                                  int dead_color_scheme, int dead_offset,
                                  int width, int height);
    private: 
        static uint32_t colors[10][256];
};

#endif //_RAINBOW_H
