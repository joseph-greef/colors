#ifndef _RAINBOW_H
#define _RAINBOW_H

extern "C" {
#include "gifenc/gifenc.h"
}
#include <SDL.h>
#include <stdint.h>

#define RAINBOW_LENGTH 256
#define GIF_COLOR_LEN 256

class Rainbows {
    public:
        Rainbows(int width, int height);
        ~Rainbows();
        void age_to_pixels(int *age_board, uint32_t *pixels); 
        void handle_input(SDL_Event event, bool control, bool shift);
        void randomize_colors();
        void start();
        void stop();
    private: 
        int alive_color_scheme_;
        int alive_offset_;
        int dead_color_scheme_;
        int dead_offset_;
        ge_GIF *gif_;

        int saved_alive_color_scheme_;
        int saved_dead_color_scheme_;

        int height_;
        int width_;

        static uint32_t colors[][RAINBOW_LENGTH];
        static int num_colors;

        void save_gif_frame(int *age_board);
        void start_gif();
};

#endif //_RAINBOW_H
