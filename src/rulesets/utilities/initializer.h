#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include <random>
#include "SDL2/SDL.h"

#include "board.cuh"

class Initializer {
private:
    Board<int> **board_ptr_;
    int density_;
    int dot_radius_;
    int word_size_;

    std::string init_words(std::string words);
public:
    //Initializer(Board *b);

    //These take pointers to boards so the initializer always operates on the
    //current board.
    Initializer(Board<int> **board, int density, int dot_radius);
    ~Initializer();

    void clear_board();
    void init_center_cross();
    void init_center_diamond();
    void init_center_square();
    void init_random_board();

    void start();
    void stop();
};

#endif //_INITIALIZER_H
