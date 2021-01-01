#ifndef _RAINBOW_H
#define _RAINBOW_H

extern "C" {
#include "gifenc.h"
}
#include <SDL2/SDL.h>
#include <stdint.h>

#define RAINBOW_LENGTH 256
#define GIF_COLOR_LEN 256

class Rainbows {
    public:
        Rainbows(int width, int height, int color_speed);
        ~Rainbows();
        void age_to_pixels(int *age_board, uint32_t *pixels); 
        void randomize_colors(bool control, bool shift);
        void reset_colors(bool control, bool shift);
        void start();
        void stop();
    private: 
        int alive_color_scheme_;
        int alive_offset_;
        bool changing_background_;
        int color_counter_;
        int color_offset_;
        int color_speed_;
        int dead_color_scheme_;
        int dead_offset_;
        ge_GIF *gif_;
        int gif_frames_;

        int saved_alive_color_scheme_;
        int saved_dead_color_scheme_;

        int height_;
        int width_;

        static uint32_t colors[][RAINBOW_LENGTH];
        static int num_colors;

        void save_gif_frame(int *age_board);
        void toggle_colors(bool control, bool shift);
        void toggle_gif(bool control, bool shift);
};

#endif //_RAINBOW_H
