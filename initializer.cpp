
#include <climits>

#include "initializer.h"
#include "input_manager.h"


Initializer::Initializer(int width, int height)
    : density_(1)
    , dot_radius_(54)
    , height_(height)
    , width_(width)
{
    InputManager::add_var_changer(&density_,    SDLK_d, 10, 0, 100, "Density");
    InputManager::add_var_changer(&dot_radius_, SDLK_s, 1, 0, INT_MAX, "Dot Size");
}

Initializer::~Initializer() {
    InputManager::remove_var_changer(SDLK_d);
    InputManager::remove_var_changer(SDLK_s);
}

//clears the board. If changing_background is true sets everything to -1
//so it will age, otherwise sets it to 0 so it won't
void Initializer::clear_board(int *board) {
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = 0;
        }
    }
}

//randomly initializes the board with density percent alive cells
void Initializer::init_board(int *board) {
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = (rand() % 100 < density_ ? 1 : 0);
        }
    }
}

//clears the board and draws a dot in the center with side length density/10
void Initializer::init_center_square(int *board) {
    clear_board(board);
    for (int i = width_ / 2 - dot_radius_; i < width_ / 2 + dot_radius_; i++) {
        for (int j = height_ /  2 - dot_radius_; j < height_ / 2 + dot_radius_; j++) {
            board[j * width_ + i] = 1;
        }
    }
}

void Initializer::init_center_diamond(int *board) {
    clear_board(board);
    for (int i = width_ / 2 - dot_radius_; i < width_ / 2 + dot_radius_; i++) {
        for (int j = height_ /  2 - dot_radius_; j < height_ / 2 + dot_radius_; j++) {
            if(abs(i - width_ / 2)+abs(j - height_ / 2) < dot_radius_) {
                board[j * width_ + i] = 1;
            }
        }
    }
}

void Initializer::init_center_cross(int *board) {
    clear_board(board);
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
}

