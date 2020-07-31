
#include "game.h"
#include "rulesets/lifelike.h"

#include <iostream>
#include <SDL.h>


Game::Game() 
    : _width(960)
    , _height(540)
{
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    _ruleset = new LifeLike(_width, _height);

    _window = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               _width,                 // width
                               _height,                // height
                               SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED);
}

Game::~Game() {
    delete _ruleset;
}

int Game::main() {
    SDL_Event event;
    bool running = true, shift = false, control = false;

    while(running) {
        _ruleset->get_pixels((uint32_t*)(SDL_GetWindowSurface(_window)->pixels));
        SDL_UpdateWindowSurface(_window);
        _ruleset->tick();
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
