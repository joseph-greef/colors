
#include <climits>
#include <iostream>

#include "font8x8.h"
#include "initializer.h"
#include "input_manager.h"


Initializer::Initializer(int **board_ptr, int density, int dot_radius, int width,
                         int height)
    : board_ptr_(board_ptr)
    , board_changed_(false)
    , density_(density)
    , dot_radius_(dot_radius)
    , height_(height)
    , width_(width)
{
}

Initializer::~Initializer() {
}

void Initializer::clear_board() {
    int *board = *board_ptr_;
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = 0;
        }
    }
    board_changed_ = true;
}

void Initializer::init_center_cross() {
    int *board = *board_ptr_;
    clear_board();
    for (int i = width_ / 2 - density_; i < width_ / 2 + density_; i++) {
        for (int j = height_ /  2 - dot_radius_; j < height_ / 2 + dot_radius_; j++) {
            board[j * width_ + i] = 1;
        }
    }
    for (int i = width_ / 2 - dot_radius_; i < width_ / 2 + dot_radius_; i++) {
        for (int j = height_ /  2 - density_; j < height_ / 2 + density_; j++) {
            board[j * width_ + i] = 1;
        }
    }
    board_changed_ = true;
}

void Initializer::init_center_diamond() {
    int *board = *board_ptr_;
    clear_board();
    for (int i = width_ / 2 - dot_radius_; i < width_ / 2 + dot_radius_; i++) {
        for (int j = height_ /  2 - dot_radius_; j < height_ / 2 + dot_radius_; j++) {
            if(abs(i - width_ / 2)+abs(j - height_ / 2) < dot_radius_) {
                board[j * width_ + i] = 1;
            }
        }
    }
    board_changed_ = true;
}

void Initializer::init_center_square() {
    int *board = *board_ptr_;
    clear_board();
    for (int i = width_ / 2 - dot_radius_; i < width_ / 2 + dot_radius_; i++) {
        for (int j = height_ /  2 - dot_radius_; j < height_ / 2 + dot_radius_; j++) {
            board[j * width_ + i] = 1;
        }
    }
    board_changed_ = true;
}

void Initializer::init_random_board() {
    int *board = *board_ptr_;
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = (rand() % 100 < density_ ? 1 : 0);
        }
    }
    board_changed_ = true;
}

std::string Initializer::init_words(std::string words) {
    int *board = *board_ptr_;
    int size = 8;
    int full_width = words.length() * 10 * size;
    int x = (width_ - full_width) / 2;
    int y = (height_ - 8 * size) / 2;

    clear_board();
    for(char c: words) {
        if(c < sizeof(font8x8) / sizeof(font8x8[0])) {
            std::cout << c;
            for(int i = 0; i < 8 * size; i++) {
                for(int j = 0; j < 8 * size; j++) {
                    if(j % size <= 1 && i % size <= 1) {
                        if(font8x8[c][j/size] & (1 << i/size)) {
                            board[(y + j) * width_ + x + i] = 1;
                        }
                    }
                }
            }


        }
        x += 10 * size;
    }
    std::cout << std::endl;
    board_changed_ = true;
    return "";
}

void Initializer::start() { 
    ADD_FUNCTION_CALLER(&Initializer::init_random_board, SDL_SCANCODE_I, false, false,
                        "(Init) Initialize random board");
    ADD_FUNCTION_CALLER(&Initializer::clear_board, SDL_SCANCODE_K, false, false,
                        "(Init) Clear board");
    ADD_FUNCTION_CALLER(&Initializer::init_center_square, SDL_SCANCODE_O, false, false,
                        "(Init) Initialize center square");
    ADD_FUNCTION_CALLER(&Initializer::init_center_diamond, SDL_SCANCODE_U, false, false,
                        "(Init) Initialize center diamond");
    ADD_FUNCTION_CALLER(&Initializer::init_center_cross, SDL_SCANCODE_Y, false, false,
                        "(Init) Initialize center cross");
    ADD_FUNCTION_CALLER_W_ARGS(&Initializer::init_words, SDL_SCANCODE_W, false, false,
                        "(Init) Initialize words on board", _1);

    InputManager::add_int_changer(&density_, SDL_SCANCODE_H,
                                  false, false, 0, 100,
                                  "(Init) Initialization density and cross width");
    InputManager::add_int_changer(&dot_radius_, SDL_SCANCODE_J,
                                  false, false, 0, INT_MAX,
                                  "(Init) Center dot radius");
}

void Initializer::stop() { 
    InputManager::remove_var_changer(SDL_SCANCODE_I, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_K, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_O, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_U, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Y, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_W, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_H, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_J, false, false);
}

bool Initializer::was_board_changed() {
    bool tmp = board_changed_;
    board_changed_ = false;
    return tmp;
}
