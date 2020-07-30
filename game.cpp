
#include "game.h"

#include <iostream>
#include <SDL.h>


Game::Game() 
    : _board(),
      _initer(&_board),
      _screen(&_board) {

    _board.set_update_algorithm(0);
    _initer.init_board();
}

Game::~Game() {
}

int Game::main() {
    SDL_Event event;
    bool running = true, shift = false, control = false;

    while(running) {
        //translate board to pixels
        _screen.draw_board();
        //and draw it
        _screen.update_window();
        //update board at the end so the first frame will get displayed
        _board.update_board();

        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                exit(0);
            }
            else if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
                switch(event.key.keysym.sym) {
                    case SDLK_LCTRL:
                    case SDLK_RCTRL:
                        control = event.type == SDL_KEYDOWN;
                        break;
                    case SDLK_LSHIFT:
                    case SDLK_RSHIFT:
                        shift = event.type == SDL_KEYDOWN;
                        break;
                    case SDLK_ESCAPE:
                        exit(0);
                }
            }
            //TODO:
            //_board.handle_input()
            //_screen.handle_input()
        }
    }

    return 0;
}
