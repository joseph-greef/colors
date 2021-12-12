
#include <climits>
#include <iostream>

#include "font8x8.h"
#include "initializer.h"
#include "input_manager.h"


Initializer::Initializer(Buffer<int> **buffer_ptr, int density, int dot_radius)
    : buffer_ptr_(buffer_ptr)
    , density_(density)
    , dot_radius_(dot_radius)
    , word_size_(8)
{
}

Initializer::~Initializer() {
}

void Initializer::clear_buffer() {
    Buffer<int> *buffer = (*buffer_ptr_);
    buffer->clear();
    buffer->copy_host_to_device();
}

void Initializer::init_center_cross() {
    Buffer<int> *buffer = *buffer_ptr_;
    int h = buffer->h_;
    int w = buffer->w_;
    buffer->clear();
    for(int i = w / 2 - density_; i < w / 2 + density_; i++) {
        for(int j = h / 2 - dot_radius_; j < h / 2 + dot_radius_; j++) {
            buffer->set(i, j, 1);
        }
    }
    for(int i = w / 2 - dot_radius_; i < w / 2 + dot_radius_; i++) {
        for(int j = h /  2 - density_; j < h / 2 + density_; j++) {
            buffer->set(i, j, 1);
        }
    }
    buffer->copy_host_to_device();
}

void Initializer::init_center_diamond() {
    Buffer<int> *buffer = *buffer_ptr_;
    int h = buffer->h_;
    int w = buffer->w_;
    buffer->clear();
    for (int i = w / 2 - dot_radius_; i < w / 2 + dot_radius_; i++) {
        for (int j = h /  2 - dot_radius_; j < h / 2 + dot_radius_; j++) {
            if(abs(i - w / 2)+abs(j - h / 2) < dot_radius_) {
                buffer->set(i, j, 1);
            }
        }
    }
    buffer->copy_host_to_device();
}

void Initializer::init_center_square() {
    Buffer<int> *buffer = *buffer_ptr_;
    int h = buffer->h_;
    int w = buffer->w_;
    buffer->clear();
    for (int i = w / 2 - dot_radius_; i < w / 2 + dot_radius_; i++) {
        for (int j = h /  2 - dot_radius_; j < h / 2 + dot_radius_; j++) {
            buffer->set(i, j, 1);
        }
    }
    buffer->copy_host_to_device();
}

void Initializer::init_random_buffer() {
    Buffer<int> *buffer = *buffer_ptr_;
    int h = buffer->h_;
    int w = buffer->w_;
    buffer->clear();
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            buffer->set(i, j, (rand() % 100 < density_ ? 1 : 0));
        }
    }
    buffer->copy_host_to_device();
}

std::string Initializer::init_words(std::string words) {
    Buffer<int> *buffer = *buffer_ptr_;
    int full_width = words.length() * 10 * word_size_;
    int x = (buffer->w_ - full_width) / 2;
    int y = (buffer->h_ - 8 * word_size_) / 2;

    buffer->clear();
    for(char c: words) {
        if(c < sizeof(font8x8) / sizeof(font8x8[0])) {
            std::cout << c;
            for(int i = 0; i < 8 * word_size_; i++) {
                for(int j = 0; j < 8 * word_size_; j++) {
                    if(j % word_size_ <= density_ && i % word_size_ <= density_) {
                        if(font8x8[c][j/word_size_] & (1 << i/word_size_)) {
                            buffer->set(x + i, y + j, 1);
                        }
                    }
                }
            }


        }
        x += 10 * word_size_;
    }
    std::cout << std::endl;
    buffer->copy_host_to_device();
    return "";
}

void Initializer::start() {
    ADD_FUNCTION_CALLER(&Initializer::init_random_buffer, SDL_SCANCODE_I, false, false,
                        "Init", "Initialize random buffer");
    ADD_FUNCTION_CALLER_W_ARGS(&Initializer::init_words, StringFunc, SDL_SCANCODE_I, true, false,
                        "Init", "Initialize words on buffer", _1);
    ADD_FUNCTION_CALLER(&Initializer::clear_buffer, SDL_SCANCODE_K, false, false,
                        "Init", "Clear buffer");
    ADD_FUNCTION_CALLER(&Initializer::init_center_square, SDL_SCANCODE_O, false, false,
                        "Init", "Initialize center square");
    ADD_FUNCTION_CALLER(&Initializer::init_center_diamond, SDL_SCANCODE_U, false, false,
                        "Init", "Initialize center diamond");
    ADD_FUNCTION_CALLER(&Initializer::init_center_cross, SDL_SCANCODE_Y, false, false,
                        "Init", "Initialize center cross");

    InputManager::add_int_changer(&density_, SDL_SCANCODE_H,
                                  false, false, 0, 100,
                                  "Init", "Initialization and word density, and cross width");
    InputManager::add_int_changer(&dot_radius_, SDL_SCANCODE_J,
                                  false, false, 0, INT_MAX,
                                  "Init", "Center dot radius");
    InputManager::add_int_changer(&word_size_, SDL_SCANCODE_I,
                                  false, true, 0, INT_MAX,
                                  "Init", "Change word size multiplier");
}

void Initializer::stop() {
    InputManager::remove_var_changer(SDL_SCANCODE_I, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_I, true, false);
    InputManager::remove_var_changer(SDL_SCANCODE_I, false, true);
    InputManager::remove_var_changer(SDL_SCANCODE_K, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_O, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_U, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Y, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_H, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_J, false, false);
}

