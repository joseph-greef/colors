
#include <ctime>
#include <climits>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "rainbows.cuh"
#include "input_manager.h"
#include "rainbows.h"

Rainbows::Rainbows(int color_speed)
    : alive_color_scheme_(1)
    , alive_offset_(0)
    , changing_background_(false)
    , color_counter_(0)
    , color_offset_(0)
    , color_speed_(color_speed)
    , dead_color_scheme_(0)
    , dead_offset_(0)
    , gif_(NULL)
    , gif_delay_(2)
    , gif_frames_(0)
    , gif_frames_setting_(0)
    , gif_loop_(true)
    , saved_alive_color_scheme_(2)
    , saved_dead_color_scheme_(2)
    , last_height_(0)
    , last_width_(0)
{
}

Rainbows::~Rainbows() {
}

void Rainbows::age_to_pixels(Board<int> *board, Board<Pixel<uint8_t>> *pixels, bool use_gpu) {
    last_height_ = board->height_;
    last_width_ = board->width_;
    if(use_gpu) {
        call_age_to_pixels_kernel(board, pixels,
                                  alive_color_scheme_, dead_color_scheme_,
                                  alive_offset_, dead_offset_, color_offset_,
                                  changing_background_);

        pixels->copy_device_to_host();
    }
    else {
        for(int index = 0; index < board->height_ * board->width_; index++) {
            age_to_pixels_step(board, pixels, index,
                               colors_host[alive_color_scheme_],
                               colors_host[dead_color_scheme_], alive_offset_,
                               dead_offset_, color_offset_, changing_background_);
        }
    }

    if(gif_) {
        save_gif_frame(board);
    }

    color_counter_++;
    if(color_speed_ > 0 && color_counter_ >= color_speed_) {
        color_counter_ = 0;
        color_offset_--;
    }
    else if (color_speed_ < 0) {
        color_offset_ += (color_speed_);
    }
}

void Rainbows::randomize_colors() {
    alive_offset_ = rand() % RAINBOW_LENGTH;
    color_offset_ = 0;
    dead_offset_ = rand() % RAINBOW_LENGTH;
}

void Rainbows::reset_colors() {
    alive_offset_ = 0;
    color_offset_ = 0;
    dead_offset_ = 0;
}

void Rainbows::save_gif_frame(Board<int> *board) {
    for(int i = 0; i < board->width_ * board->height_; i++) {
        if(board->get(i) > 0) {
            gif_->frame[i] = (board->get(i) + alive_offset_ + color_offset_) &
                             255;
        }
        if(board->get(i) < 0) {
            gif_->frame[i] = (-board->get(i) + dead_offset_ + color_offset_) &
                             255;
        }
        else {
            gif_->frame[i] = (-board->get(i) + dead_offset_) & 255;
        }
    }
    ge_add_frame(gif_, gif_delay_);

    gif_frames_--;
    if(gif_frames_ == 0) {
        ge_close_gif(gif_);
        gif_ = NULL;
    }
}

void Rainbows::start() {
    InputManager::add_bool_toggler(&changing_background_, SDL_SCANCODE_B,
                                   false, false,
                                   "Rainbows", "Toggle Changing Background");
    InputManager::add_bool_toggler(&gif_loop_, SDL_SCANCODE_BACKSLASH,
                                   true, true,
                                   "Rainbows", "Toggle looping gif");

    ADD_FUNCTION_CALLER(&Rainbows::toggle_colors, SDL_SCANCODE_C, false, false,
                        "Rainbows", "Toggle colors");
    ADD_FUNCTION_CALLER(&Rainbows::reset_colors, SDL_SCANCODE_L, false, false,
                        "Rainbows", "Reset colors");
    ADD_FUNCTION_CALLER(&Rainbows::randomize_colors, SDL_SCANCODE_GRAVE, false, false,
                        "Rainbows", "Randomize colors");
    ADD_FUNCTION_CALLER(&Rainbows::toggle_gif, SDL_SCANCODE_BACKSLASH, false, false,
                        "Rainbows", "Toggle gif recording");

    InputManager::add_int_changer(&dead_color_scheme_,  SDL_SCANCODE_M, false, false,
                                  0, Rainbows::num_colors-1, "Rainbows", "Dead scheme");
    InputManager::add_int_changer(&alive_color_scheme_, SDL_SCANCODE_N, false, false,
                                  0, Rainbows::num_colors-1, "Rainbows", "Alive scheme");

    InputManager::add_int_changer(&gif_delay_,  SDL_SCANCODE_BACKSLASH, false, true,
                                  2, INT_MAX, "Rainbows", "Gif delay (sec/100)");
    InputManager::add_int_changer(&gif_frames_setting_,  SDL_SCANCODE_BACKSLASH,
                                  true, false, 0, INT_MAX,
                                  "Rainbows", "Number of Gif frames to record");

    InputManager::add_int_changer(&dead_offset_,  SDL_SCANCODE_COMMA, false, false,
                                  INT_MIN, INT_MAX, "Rainbows", "Dead offset");
    InputManager::add_int_changer(&alive_offset_, SDL_SCANCODE_PERIOD, false, false,
                                  INT_MIN, INT_MAX, "Rainbows", "Alive offset");

    InputManager::add_int_changer(&color_speed_, SDL_SCANCODE_SLASH, false, false,
                                  INT_MIN, INT_MAX, "Rainbows", "Color speed");
}

void Rainbows::stop() {
    InputManager::remove_var_changer(SDL_SCANCODE_B, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_GRAVE, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_BACKSLASH, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_C, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_L, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_M, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_N, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_BACKSLASH, true, false);
    InputManager::remove_var_changer(SDL_SCANCODE_BACKSLASH, false, true);
    InputManager::remove_var_changer(SDL_SCANCODE_BACKSLASH, true, true);
    InputManager::remove_var_changer(SDL_SCANCODE_COMMA, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_PERIOD, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_SLASH, false, false);
}

void Rainbows::toggle_colors() {
    int tmp_alive = alive_color_scheme_;
    int tmp_dead = dead_color_scheme_;
    alive_color_scheme_ = saved_alive_color_scheme_;
    dead_color_scheme_ = saved_dead_color_scheme_;
    saved_alive_color_scheme_ = tmp_alive;
    saved_dead_color_scheme_ = tmp_dead;
}

void Rainbows::toggle_gif() {
    if(gif_) {
        ge_close_gif(gif_);
        gif_ = NULL;
    }
    else {
        static uint8_t rainbow_no_alpha[GIF_COLOR_LEN * 3] = { 0 };
        std::time_t t = std::time(nullptr);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.gif");
        std::string str = oss.str();

        for(int i = 0; i < GIF_COLOR_LEN; i++) {
            int nai = 3 * i;
            uint32_t color = colors_host[alive_color_scheme_][i];
            uint8_t *components = (uint8_t*)&color;
            rainbow_no_alpha[nai] = components[2];
            rainbow_no_alpha[nai + 1] = components[1];
            rainbow_no_alpha[nai + 2] = components[0];
        }

        gif_ = ge_new_gif(str.c_str(), last_width_, last_height_,
                          rainbow_no_alpha, 8, gif_loop_ ? 0 : -1);
        gif_frames_ = gif_frames_setting_;
    }
}

uint32_t Rainbows::colors_host[][RAINBOW_LENGTH] = {
#include "gradients.h"
};

int Rainbows::num_colors = sizeof(colors_host) / sizeof(colors_host[0]);

