#ifndef _RAINBOW_H
#define _RAINBOW_H

#include <stdint.h>

#define RAINBOW_LENGTH 256
class Rainbows {
    public:
        static void age_to_pixels(int *age_board, uint32_t *pixels, 
                                  int alive_color_scheme, int alive_offset,
                                  int dead_color_scheme, int dead_offset,
                                  int width, int height);
        static void age_to_bw_pixels(int *age_board, uint32_t *pixels, 
                                     int width, int height);
    private: 
        static uint32_t colors[10][RAINBOW_LENGTH];
};

#endif //_RAINBOW_H
