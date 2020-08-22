#ifndef _INITIALIZER_H
#define _INITIALIZER_H


#include "info.h"

#include <random>

class Initializer {
private:
    int height_;
    int width_;
    
public:
    //Initializer(Board *b);
    Initializer(int width, int height);
    ~Initializer();

    void clear_board(int* board);
    void init_board(int *board, int density);
    void init_center_cross(int *board, int line_width, int radius);
    void init_center_diamond(int *board, int radius);
    void init_center_square(int *board, int radius);

};

#endif //_INITIALIZER_H
