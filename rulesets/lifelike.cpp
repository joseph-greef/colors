
#include <climits>
#include <iostream>
#include <stdlib.h>

#include "lifelike.h"

LifeLike::LifeLike(int width, int height)
    : Ruleset(width, height)
    , alive_color_scheme_(0)
    , alive_offset_(128)
    , dead_color_scheme_(0)
    , dead_offset_(0)
    , density_(1)
    , dot_radius_(15)
    , draw_color_(false)
    , initializer_(width, height)
    , num_faders_(0)
    , rainbows_()
{
    //Random pretty pattern
    //bool born_tmp[9] = {0, 0, 0, 0, 1, 1 ,1 ,1, 1};
    //bool stay_alive_tmp[9] = {1, 0, 0, 1, 1, 1 ,1 ,1, 0};
    //star wars
    bool born_tmp[9] = {0, 0, 1, 0, 0, 0 ,0 ,0, 0};
    bool stay_alive_tmp[9] = {0, 0, 0, 1, 1, 1, 0, 0, 0};
    num_faders_ = 4;

    memcpy(born_, born_tmp, sizeof(born_));
    memcpy(stay_alive_, stay_alive_tmp, sizeof(stay_alive_));

    board_ = new int[width*height];
    board_buffer_ = new int[width*height];

    initializer_.init_center_cross(board_, dot_radius_, density_);

    add_var_changer(&density_, SDLK_d, 10, 0, 100, "Density");
    add_var_changer(&num_faders_, SDLK_f, 1, 0, INT_MAX, "Num Faders");
    add_var_changer(&dead_color_scheme_, SDLK_m, 1, 0, 9, "Dead Scheme");
    add_var_changer(&alive_color_scheme_, SDLK_n, 1, 0, 9, "Alive Scheme");
    add_var_changer(&dot_radius_, SDLK_s, 1, 0, INT_MAX, "Dot Size");
}

LifeLike::~LifeLike() {
    delete board_;
    delete board_buffer_;
}

void LifeLike::get_pixels(uint32_t *pixels) {
    if(draw_color_) {
        Rainbows::age_to_pixels(board_, pixels,
                                alive_color_scheme_, alive_offset_,
                                dead_color_scheme_, dead_offset_,
                                width_, height_);
    }
    else {
        Rainbows::age_to_bw_pixels(board_, pixels,
                                   width_, height_);
    }
}

void LifeLike::handle_input(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_c:
                draw_color_ = !draw_color_;
                break;
            case SDLK_e:
                initializer_.init_center_square(board_, dot_radius_);
                break;
            case SDLK_i:
                initializer_.init_board(board_, density_);
                break;
            case SDLK_r:
                randomize_ruleset();
                break;
            case SDLK_w:
                initializer_.init_center_diamond(board_, dot_radius_);
                break;
            case SDLK_x:
                initializer_.init_center_cross(board_, dot_radius_, density_);
                break;

         }
    }
    else if(event.type == SDL_KEYUP) {
        switch(event.key.keysym.sym) {
        }
    }
    handle_var_changers(event, control, shift);
}

void LifeLike::print_rules() {
    std::cout << "Born: ";
    for(int i = 0; i < 9; i++) {
        std::cout << born_[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Stay Alive: ";
    for(int i = 0; i < 9; i++) {
        std::cout << stay_alive_[i] << " ";
    }
    std::cout << std::endl;
}

void LifeLike::randomize_ruleset() {
    for(int i = 0; i < 9; i++) {
        born_[i] = (rand()%100>20 ? 1 : 0);
        stay_alive_[i] = (rand()%100>20 ? 1 : 0);
    }
    born_[0] = false;

    alive_offset_ = rand() % RAINBOW_LENGTH;
    dead_offset_ = rand() % RAINBOW_LENGTH;
}

void LifeLike::tick() {
    for(int j = 0; j < height_; j++) {
        for(int i = 0; i < width_; i++) {
            //get how many alive neighbors it has
            int neighbors = Ruleset::get_num_alive_neighbors(board_, i, j, 1,
                                                             Moore);
            int offset = j * width_ + i;

            if(board_[offset] > 0) {
                if(stay_alive_[neighbors]) {
                    board_buffer_[offset] = board_[offset] + 1;
                }
                else {
                    board_buffer_[offset] = -1;
                }

            }
            else if(board_[offset] <= -num_faders_) {
                if(born_[neighbors]) {
                    board_buffer_[offset] = 1;
                }
                else if(board_[offset] == 0) {
                    board_buffer_[offset] = 0;
                }
                else {
                    board_buffer_[offset] = board_[offset] - 1;
                }
            }
            else {
                board_buffer_[offset] = board_[offset] - 1;
            }
        }
    }

    {
        int *tmp = board_buffer_;
        board_buffer_ = board_;
        board_ = tmp;
    }

}

