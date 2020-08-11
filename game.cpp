
#include <ctime>
#include <iomanip>
#include <iostream>
#include <SDL_image.h>
#include <sstream>

#include "game.h"
#include "rulesets/lifelike.h"


Game::Game() 
    : width_(1920)
    , height_(1080)
{
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    ruleset_ = new LifeLike(width_, height_);

    window_ = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               width_,                 // width
                               height_,                // height
                               SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED);
}

Game::~Game() {
    delete ruleset_;
}

void Game::handle_input(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_p:
                ruleset_->print_rules();
                break;
            case SDLK_LEFTBRACKET: {
                //Get the time and convert it to a string.png
                std::time_t t = std::time(nullptr);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.png");
                std::string str = oss.str();

                IMG_SavePNG(SDL_GetWindowSurface(window_), str.c_str());
                break;
            }
            case SDLK_f:
                ruleset_->toggle_gpu();
                break;
        }
    }
}

int Game::main() {
    SDL_Event event;
    bool running = true, shift = false, control = false;

    //TODO: Scheduler
    while(running) {
        ruleset_->get_pixels((uint32_t*)(SDL_GetWindowSurface(window_)->pixels));
        SDL_UpdateWindowSurface(window_);
        ruleset_->tick();
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
            this->handle_input(event, control, shift);
            ruleset_->handle_input(event, control, shift);
        }
    }

    return 0;
}
