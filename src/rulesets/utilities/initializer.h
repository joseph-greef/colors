#ifndef _INITIALIZER_H
#define _INITIALIZER_H

#include <random>
#include "SDL2/SDL.h"

#include "buffer.cuh"


class Initializer {
private:
    Buffer<int> **buffer_ptr_;
    int density_;
    int dot_radius_;
    int word_size_;

    std::string init_words(std::string words);
public:
    //Initializer(Buffer *b);

    //These take pointers to buffers so the initializer always operates on the
    //current buffer.
    Initializer(Buffer<int> **buffer, int density, int dot_radius);
    ~Initializer();

    void clear_buffer();
    void init_center_cross();
    void init_center_diamond();
    void init_center_square();
    void init_random_buffer();

    void start();
    void stop();
};

#endif //_INITIALIZER_H
