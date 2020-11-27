#include <ctime>
#include <iomanip>
#include <iostream>
#include <SDL_image.h>
#include <sstream>

#include "game.h"
#include "input_manager.h"


Game::Game(int width, int height) 
    : active_ruleset_(NULL)
    , current_ruleset_(0)
    , last_ruleset_(0)
    , lock_cursor_(false)
    , width_(width)
    , height_(height)
{
    memset(rulesets_, 0, sizeof(rulesets_));
    active_ruleset_ = rulesets_[0] = new LifeLike(width_, height_);
    active_ruleset_->start();
    InputManager::add_var_changer(&current_ruleset_, SDLK_z, 0, NUM_RULESETS-1, "Ruleset");
}

Game::~Game() {
    for(int i = 0; i < NUM_RULESETS; i++) {
        if(rulesets_[i] != NULL) {
            delete rulesets_[i];
        }
    }
}

void Game::change_ruleset(int new_ruleset) {
    if(rulesets_[new_ruleset] == NULL) {
        switch(new_ruleset) {
            case 0: rulesets_[new_ruleset] = new LifeLike(width_, height_); break;
            case 1: rulesets_[new_ruleset] = new Hodge(width_, height_); break;
        }
    }
    active_ruleset_->stop();
    InputManager::reset();
    active_ruleset_ = rulesets_[new_ruleset];
    active_ruleset_->start();
}

void Game::draw_board(uint32_t *board) {
    active_ruleset_->get_pixels(board);
}

void Game::handle_input(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        switch(event.key.keysym.sym) {
            case SDLK_f:
                active_ruleset_->toggle_gpu();
                break;
            case SDLK_p:
                //ruleset_->print_rules();
                break;
        }
    }
    active_ruleset_->handle_input(event, control, shift);
}

void Game::tick(void) {
    if(last_ruleset_ != current_ruleset_) {
        change_ruleset(current_ruleset_);
    }
    last_ruleset_ = current_ruleset_;

    active_ruleset_->tick();

    return;
}

