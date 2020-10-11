#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include <random>
#include <SDL.h>

class Initializer {
private:
    int density_;
    int dot_radius_;
    int height_;
    int width_;
    
public:
    //Initializer(Board *b);
    Initializer(int density, int dot_radius, int width, int height);
    ~Initializer();

    void clear_board(int* board);
    void init_board(int *board);
    void init_center_cross(int *board);
    void init_center_diamond(int *board);
    void init_center_square(int *board);
};

#endif //_INITIALIZER_H
