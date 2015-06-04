
#include "screen.h"


Screen::Screen(Board *new_board) {
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
                                           SCREEN_WIDTH,               // width, in pixels
                                           SCREEN_HEIGHT,              // height, in pixels
                                           SDL_WINDOW_BORDERLESS          // flags
                                           );
    //SDL_GetWindowSize(window, &screen_width, &screen_height);


    screen = SDL_GetWindowSurface(window);
    pixels = (Uint32*)screen->pixels;

    draw_colors = false;
    draw_smooth = false;
    alive_offset = 0;
    dead_offset = 0;
    color_speed_divisor = 1;

    board_obj = new_board;

}

Screen::~Screen() {
    // clean up
    SDL_DestroyWindow(window);

    SDL_Quit();
}


//this function sets the pixels at (x,y) on the board to color accounting for the fact
//that cells are not always 1x1 pixels
void Screen::set_pixel(int x, int y, Uint32 color) {
    //just sets the correct block of pixels to the color
    for(int i = x*PIXELS_PER_CELL; i < x*PIXELS_PER_CELL+PIXELS_PER_CELL; i++) {
        for(int j = y*PIXELS_PER_CELL; j < y*PIXELS_PER_CELL+PIXELS_PER_CELL; j++) {
            pixels[j*SCREEN_WIDTH+i] = color;
        }
    }
}


//this function converts a board of cells into a screen's worth of pixels
void Screen::draw_board() {
    static Uint32 colors[256] = {0x000000, 0x070000, 0x0F0000, 0x170000, 0x1F0000, 0x270000, 0x2F0000, 0x370000, 0x3F0000, 0x470000, 0x4F0000, 0x570000, 0x5F0000, 0x670000, 0x6F0000, 0x770000, 0x7F0000, 0x870000, 0x8F0000, 0x970000, 0x9F0000, 0xA70000, 0xAF0000, 0xB70000, 0xBF0000, 0xC70000, 0xCF0000, 0xD70000, 0xDF0000, 0xE70000, 0xEF0000, 0xF70000, 0xFF0000, 0xFF0007, 0xFF000F, 0xFF0017, 0xFF001F, 0xFF0027, 0xFF002F, 0xFF0037, 0xFF003F, 0xFF0047, 0xFF004F, 0xFF0057, 0xFF005F, 0xFF0067, 0xFF006F, 0xFF0077, 0xFF007F, 0xFF0087, 0xFF008F, 0xFF0097, 0xFF009F, 0xFF00A7, 0xFF00AF, 0xFF00B7, 0xFF00BF, 0xFF00C7, 0xFF00CF, 0xFF00D7, 0xFF00DF, 0xFF00E7, 0xFF00EF, 0xFF00F7, 0xFF00FF, 0xF700FF, 0xEF00FF, 0xE700FF, 0xDF00FF, 0xD700FF, 0xCF00FF, 0xC700FF, 0xBF00FF, 0xB700FF, 0xAF00FF, 0xA700FF, 0x9F00FF, 0x9700FF, 0x8F00FF, 0x8700FF, 0x7F00FF, 0x7700FF, 0x6F00FF, 0x6700FF, 0x5F00FF, 0x5700FF, 0x4F00FF, 0x4700FF, 0x3F00FF, 0x3700FF, 0x2F00FF, 0x2700FF, 0x1F00FF, 0x1700FF, 0x0F00FF, 0x0700FF, 0x0000FF, 0x0007FF, 0x000FFF, 0x0017FF, 0x001FFF, 0x0027FF, 0x002FFF, 0x0037FF, 0x003FFF, 0x0047FF, 0x004FFF, 0x0057FF, 0x005FFF, 0x0067FF, 0x006FFF, 0x0077FF, 0x007FFF, 0x0087FF, 0x008FFF, 0x0097FF, 0x009FFF, 0x00A7FF, 0x00AFFF, 0x00B7FF, 0x00BFFF, 0x00C7FF, 0x00CFFF, 0x00D7FF, 0x00DFFF, 0x00E7FF, 0x00EFFF, 0x00F7FF, 0x00FFFF, 0x00FFF7, 0x00FFEF, 0x00FFE7, 0x00FFDF, 0x00FFD7, 0x00FFCF, 0x00FFC7, 0x00FFBF, 0x00FFB7, 0x00FFAF, 0x00FFA7, 0x00FF9F, 0x00FF97, 0x00FF8F, 0x00FF87, 0x00FF7F, 0x00FF77, 0x00FF6F, 0x00FF67, 0x00FF5F, 0x00FF57, 0x00FF4F, 0x00FF47, 0x00FF3F, 0x00FF37, 0x00FF2F, 0x00FF27, 0x00FF1F, 0x00FF17, 0x00FF0F, 0x00FF07, 0x00FF00, 0x07FF00, 0x0FFF00, 0x17FF00, 0x1FFF00, 0x27FF00, 0x2FFF00, 0x37FF00, 0x3FFF00, 0x47FF00, 0x4FFF00, 0x57FF00, 0x5FFF00, 0x67FF00, 0x6FFF00, 0x77FF00, 0x7FFF00, 0x87FF00, 0x8FFF00, 0x97FF00, 0x9FFF00, 0xA7FF00, 0xAFFF00, 0xB7FF00, 0xBFFF00, 0xC7FF00, 0xCFFF00, 0xD7FF00, 0xDFFF00, 0xE7FF00, 0xEFFF00, 0xF7FF00, 0xFFFF00, 0xFFFB00, 0xFFF800, 0xFFF400, 0xFFF100, 0xFFED00, 0xFFEA00, 0xFFE700, 0xFFE300, 0xFFE000, 0xFFDC00, 0xFFD901, 0xFFD601, 0xFFD201, 0xFFCF01, 0xFFCB01, 0xFFC801, 0xFFC501, 0xFFC101, 0xFFBE01, 0xFFBA01, 0xFFB701, 0xFFB402, 0xFFB002, 0xFFAD02, 0xFFA902, 0xFFA602, 0xFFA302, 0xFF9F02, 0xFF9C02, 0xFF9802, 0xFF9502, 0xFF9203, 0xF68D02, 0xEE8802, 0xE68302, 0xDE7F02, 0xD57A02, 0xCD7502, 0xC57102, 0xBD6C02, 0xB46702, 0xAC6202, 0xA45E01, 0x9C5901, 0x945401, 0x8B5001, 0x834B01, 0x7B4601, 0x734101, 0x6A3D01, 0x623801, 0x5A3301, 0x522F00, 0x4A2A00, 0x412500, 0x392000, 0x311C00, 0x291700, 0x201200, 0x180E00, 0x100900, 0x080400};

    int *board = board_obj->get_board();
    float *board_float = board_obj->get_board_float();
    if(draw_smooth) {
        int color;
        if(draw_colors) {
            for(int j = 0; j < CELL_HEIGHT; j++) {
                for(int i = 0; i < CELL_WIDTH; i++) {
                    //if we're alive modulo the age of the cell by the number of colors to get which color to draw and draw it
                    if(board[j*CELL_WIDTH+i] > 0)
                        set_pixel(i, j, colors[((board[j*CELL_WIDTH+i]+alive_offset)/color_speed_divisor) % 256]);
                    else //do the same if we're dead, but with negative age instead
                        set_pixel(i, j, colors[((-board[j*CELL_WIDTH+i]+dead_offset)/color_speed_divisor) % 256]);
                }
            }
        }
        else {
            for(int j = 0; j < CELL_HEIGHT; j++) {
                for(int i = 0; i < CELL_WIDTH; i++) {
                    color = 255 * board_float[j*CELL_WIDTH+i];
                    set_pixel(i, j, color | color << 8 | color << 16);
                }
            }
        }
    }
    else {
        //if we're drawing in color
        if(draw_colors) {
            //go over the entire board
            for(int j = 0; j < CELL_HEIGHT; j++) {
                for(int i = 0; i < CELL_WIDTH; i++) {
                    //if we're alive modulo the age of the cell by the number of colors to get which color to draw and draw it
                    if(board[j*CELL_WIDTH+i] > 0)
                        set_pixel(i, j, colors[((board[j*CELL_WIDTH+i]+alive_offset)/color_speed_divisor) % 256]);
                    else //do the same if we're dead, but with negative age instead
                        set_pixel(i, j, colors[((-board[j*CELL_WIDTH+i]+dead_offset)/color_speed_divisor) % 256]);
                }
            }
        }
        //if we're doing black and white
        else {
            //iterate over the whole board
            for(int j = 0; j < CELL_HEIGHT; j++) {
                for(int i = 0; i < CELL_WIDTH; i++) {
                    //and draw white if alive
                    if(board[j*CELL_WIDTH+i] > 0)
                        set_pixel(i, j, 0xFFFFFF);
                    //and black if dead
                    else
                        set_pixel(i, j, 0x000000);
                }
            }
        }
    }

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

void Screen::update_window() {
    SDL_UpdateWindowSurface(window);

}

