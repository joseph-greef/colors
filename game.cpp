
#include <ctime>
#include <iomanip>
#include <iostream>
#include <SDL_image.h>
#include <sstream>

#include "game.h"
#include "input_manager.h"

#define NUM_RULESETS 2
#include "rulesets/hodge.h"
#include "rulesets/lifelike.h"


Game::Game() 
    : current_ruleset_(1)
    , lock_cursor_(false)
    , ruleset_(NULL)
    , width_(1080)
    , height_(1080)
{
    static uint8_t data = 0;
    static uint8_t mask = 0;
    if(SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }
    SDL_Cursor *cursor = SDL_CreateCursor(&data, &mask, 1, 1, 0, 0);

    window_ = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               width_,                 // width
                               height_,                // height
                               SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED);
    SDL_SetCursor(cursor);

    change_ruleset(current_ruleset_);
    InputManager::add_var_changer(&current_ruleset_, SDLK_z, 1, 0, NUM_RULESETS-1, "Ruleset");
}

Game::~Game() {
    delete ruleset_;
}

void Game::change_ruleset(int new_ruleset) {
    if(ruleset_) {
        delete ruleset_;
    }

    switch(new_ruleset) {
        case 0: ruleset_ = new LifeLike(width_, height_); break;
        case 1: ruleset_ = new Hodge(width_, height_); break;
    }
}

void Game::handle_input(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_f:
                ruleset_->toggle_gpu();
                break;
            case SDLK_l:
                lock_cursor_ = !lock_cursor_;
                break;
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
        }
    }
}

int Game::main() {
    SDL_Event event;
    bool running = true, shift = false, control = false;
    int last_ruleset = current_ruleset_;

    while(running) {
        if(last_ruleset != current_ruleset_) {
            change_ruleset(current_ruleset_);
        }
        last_ruleset = current_ruleset_;

        ruleset_->get_pixels((uint32_t*)(SDL_GetWindowSurface(window_)->pixels));
        SDL_UpdateWindowSurface(window_);

        ruleset_->tick();

        while(SDL_PollEvent(&event)) {
            if(lock_cursor_) {
                SDL_WarpMouseInWindow(window_, width_/2, height_/2);
            }
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
            InputManager::handle_input(event, control, shift);
            ruleset_->handle_input(event, control, shift);
        }
    }

    return 0;
}
