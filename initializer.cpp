
#include "initializer.h"


Initializer::Initializer(int width, int height)
    : height_(height)
    , width_(width)
{
}

Initializer::~Initializer() {

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
void Initializer::init_board(int *board, int density) {
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = (rand() % 100 < density ? 1 : -1);
        }
    }
}

//clears the board and draws a dot in the center with side length density/10
void Initializer::init_center_square(int *board, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            board[j * width_ + i] = 1;
        }
    }
}

void Initializer::init_center_diamond(int *board, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            if(abs(i - width_ / 2)+abs(j - height_ / 2) < radius) {
                board[j * width_ + i] = 1;
            }
        }
    }
}


void Initializer::init_center_cross(int *board, int line_width, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - line_width; i < width_ / 2 + line_width; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            board[j * width_ + i] = 1;
        }
    }
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - line_width; j < height_ / 2 + line_width; j++) {
            board[j * width_ + i] = 1;
        }
    }
}

