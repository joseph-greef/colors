#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include <random>
#include "SDL2/SDL.h"

class Initializer {
private:
    int **board_ptr_;
    bool board_changed_;
    int density_;
    int dot_radius_;
    int height_;
    int width_;
    int word_size_;

    std::string init_words(std::string words);
public:
    //Initializer(Board *b);
    Initializer(int **board, int density, int dot_radius, int width, int height);
    ~Initializer();

    void clear_board();
    void init_center_cross();
    void init_center_diamond();
    void init_center_square();
    void init_random_board();

    bool was_board_changed();

    void start();
    void stop();
};

#endif //_INITIALIZER_H
