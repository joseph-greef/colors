
#include "screen.h"
#include "rulesets/rainbows.h"


Screen::Screen(Board *new_board) {
    board_obj = new_board;

    SDL_Surface* screen = NULL;
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0)
    {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    // create a window
    window = SDL_CreateWindow(
                                           "SDL 2 window",             // window title
                                           SDL_WINDOWPOS_CENTERED,     // x position, centered
                                           SDL_WINDOWPOS_CENTERED,     // y position, centered
                                           960,               // width, in pixels
                                           540,              // height, in pixels
                                           SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED          // flags
                                           );
    //SDL_GetWindowSize(window, &screen_width, &screen_height);

    int x, y;
    SDL_GetWindowSize(window, &x, &y);

    board_obj->set_cell_height(y);
    board_obj->set_cell_width(x);

    screen = SDL_GetWindowSurface(window);
    pixels = (Uint32*)screen->pixels;

    draw_colors = false;
    draw_smooth = false;
    alive_offset = 0;
    dead_offset = 0;
    color_speed_divisor = 1;

    color_scheme = 0;


}

Screen::~Screen() {
    // clean up
    SDL_DestroyWindow(window);

    SDL_Quit();
}


//this function sets the pixels at (x,y) on the board to color accounting for the fact
//that cells are not always 1x1 pixels
void Screen::set_pixel(int x, int y, Uint32 color) {
    pixels[y*board_obj->get_cell_width()+x] = color;
    //just sets the correct block of pixels to the color
    //for(int i = x*PIXELS_PER_CELL; i < x*PIXELS_PER_CELL+PIXELS_PER_CELL; i++) {
    //    for(int j = y*PIXELS_PER_CELL; j < y*PIXELS_PER_CELL+PIXELS_PER_CELL; j++) {
    //        pixels[j*board_obj->get_cell_width()+i] = color;
    //    }
    //}
}

SDL_Point Screen::get_screen_dimensions() {
    SDL_Point p = {960, 540};
    return p;
}

//this function converts a board of cells into a screen's worth of pixels
void Screen::draw_board() {
    int *board = board_obj->get_board();
    float *board_float = board_obj->get_board_float();
    int width = board_obj->get_cell_width();
    int height = board_obj->get_cell_height();
    pixels = (Uint32*)(SDL_GetWindowSurface(window)->pixels);

    if(draw_smooth) {
        int color;
        if(draw_colors) {
            for(int j = 0; j < height; j++) {
                for(int i = 0; i < width; i++) {
                    //if we're alive modulo the age of the cell by the number of colors to get which color to draw and draw it
                    if(board[j*width+i] > 0)
                        set_pixel(i, j, old_colors[color_scheme][((board[j*width+i]+alive_offset)/color_speed_divisor) & 255]);
                    else //do the same if we're dead, but with negative age instead
                        set_pixel(i, j, old_colors[color_scheme][((-board[j*width+i]+dead_offset)/color_speed_divisor) & 255]);
                }
            }
        }
        else {
            for(int j = 0; j < height; j++) {
                for(int i = 0; i < width; i++) {
                    color = 255 * board_float[j*width+i];
                    set_pixel(i, j, color | color << 8 | color << 16);
                }
            }
        }
    }
    else {
        //if we're drawing in color
        if(draw_colors) {
            //go over the entire board
            for(int j = 0; j < height; j++) {
                for(int i = 0; i < width; i++) {
                    //if we're alive modulo the age of the cell by the number of colors to get which color to draw and draw it
                    if(board[j*width+i] > 0)
                        set_pixel(i, j, old_colors[color_scheme][((board[j*width+i]+alive_offset)) & 255]);
                    else //do the same if we're dead, but with negative age instead
                        set_pixel(i, j, old_colors[color_scheme][((-board[j*width+i]+dead_offset)) & 255]);
                }
            }
        }
        //if we're doing black and white
        else {
            //iterate over the whole board
            for(int j = 0; j < height; j++) {
                for(int i = 0; i < width; i++) {
                    //and draw white if alive
                    if(board[j*width+i] > 0)
                        set_pixel(i, j, 0xFFFFFF);
                    //and black if dead
                    else
                        set_pixel(i, j, 0x000000);
                }
            }
        }
    }

    SDL_UpdateWindowSurface(window);

}

void Screen::set_color_scheme(int scheme) {
    color_scheme = scheme;
}

//randomly generates the color offsets and ruleset.
void Screen::reset_colors() {
    dead_offset = rand() % 256;
    alive_offset = rand() % 256;

}


void Screen::flip_draw_colors() {
    draw_colors = !draw_colors;
}

void Screen::flip_draw_smooth() {
    draw_smooth = !draw_smooth;
}


void Screen::set_color_speed_divisor(uint8_t new_color_speed_divisor) {
    color_speed_divisor = new_color_speed_divisor;
}

