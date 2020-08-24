#ifndef _RAINBOW_H
#define _RAINBOW_H

#include <stdint.h>

#define RAINBOW_LENGTH 256
class Rainbows {
    public:
        Rainbows(int width, int height);
        ~Rainbows();
        void age_to_pixels(int *age_board, uint32_t *pixels); 
        void handle_input(SDL_Event event, bool control, bool shift);
        void randomize_colors();
    private: 
        int alive_color_scheme_;
        int alive_offset_;
        int dead_color_scheme_;
        int dead_offset_;
        int height_;
        int width_;

        static uint32_t colors[][RAINBOW_LENGTH];
        static int num_colors;
};

#endif //_RAINBOW_H
