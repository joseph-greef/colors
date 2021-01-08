
#include <climits>

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

    //InputManager::add_int_changer(&density_,    SDLK_h, 0, 100, "(Init) Density");
    //InputManager::add_int_changer(&dot_radius_, SDLK_j, 0, INT_MAX, "(Init) Dot Size");
}

void Initializer::stop() { 
    InputManager::remove_var_changer(SDL_SCANCODE_I, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_K, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_O, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_U, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Y, false, false);

    //InputManager::remove_var_changer(SDLK_h);
    //InputManager::remove_var_changer(SDLK_j);
}

bool Initializer::was_board_changed() {
    bool tmp = board_changed_;
    board_changed_ = false;
    return tmp;
}
