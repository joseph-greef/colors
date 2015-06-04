#ifndef SCREEN_H_INCLUDED
#define SCREEN_H_INCLUDED

#include <SDL.h>
#include <stdlib.h>
#include <iostream>

#include "info.h"
#include "board.h"



class Screen {
    private:
        bool draw_colors, draw_smooth;
        int alive_offset;
        int dead_offset;
        uint8_t color_speed_divisor;
        Uint32 *pixels;
        SDL_Window *window;
        Board *board_obj;
    public:
        Screen(Board *new_board);
        ~Screen();
        void set_pixel(int x, int y, Uint32 color);
        void draw_board();
        void reset_colors();
        void flip_draw_colors();
        void flip_draw_smooth();
        void set_color_speed_divisor(uint8_t new_color_speed_divisor);
        void update_window();
};



#endif // SCREEN_H_INCLUDED
