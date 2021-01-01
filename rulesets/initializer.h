#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include <random>
#include <SDL2/SDL.h>

class Initializer {
private:
    int **board_ptr_;
    bool board_changed_;
    int density_;
    int dot_radius_;
    int height_;
    int width_;
    
public:
    //Initializer(Board *b);
    Initializer(int **board, int density, int dot_radius, int width, int height);
    ~Initializer();

    void clear_board(bool control, bool shift);
    void init_center_cross(bool control, bool shift);
    void init_center_diamond(bool control, bool shift);
    void init_center_square(bool control, bool shift);
    void init_random_board(bool control, bool shift);

    bool was_board_changed();

    void start();
    void stop();
};

#endif //_INITIALIZER_H
