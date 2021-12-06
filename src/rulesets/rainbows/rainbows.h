#ifndef _RAINBOW_H
#define _RAINBOW_H

extern "C" {
#include "gifenc.h"
}
#include <SDL2/SDL.h>
#include <stdint.h>

#include "board.cuh"

#define RAINBOW_LENGTH 256
#define GIF_COLOR_LEN 256

class Rainbows {
    public:
        Rainbows(int color_speed);
        ~Rainbows();
        void age_to_pixels(Board<int> *board, uint32_t *pixels, bool use_gpu);
        void randomize_colors();
        void reset_colors();
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
        int gif_delay_;
        int gif_frames_setting_;
        int gif_frames_;
        bool gif_loop_;

        int saved_alive_color_scheme_;
        int saved_dead_color_scheme_;

        int last_height_;
        int last_width_;

        static uint32_t colors[][RAINBOW_LENGTH];
        static int num_colors;

        void save_gif_frame(Board<int> *board);
        void toggle_colors();
        void toggle_gif();
};

#endif //_RAINBOW_H
