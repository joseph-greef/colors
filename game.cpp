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
    , last_ruleset_(current_ruleset_)
    , lock_cursor_(false)
    , width_(width)
    , height_(height)
{
    rulesets_.push_back(new LifeLike(width_, height_));
    rulesets_.push_back(new Hodge(width_, height_));
    rulesets_.push_back(new Ants(width_, height_));

    active_ruleset_ = rulesets_[current_ruleset_];
    active_ruleset_->start();

    InputManager::add_int_changer(&current_ruleset_, SDLK_z, 0, NUM_RULESETS-1, "(Game) Ruleset");
}

Game::~Game() {
    active_ruleset_->stop();
    for(Ruleset *ruleset: rulesets_) {
        delete ruleset;
    }
    InputManager::remove_var_changer(SDLK_z);
}

void Game::change_ruleset(int new_ruleset) {
    active_ruleset_->stop();
    InputManager::reset();
    active_ruleset_ = rulesets_[new_ruleset];
    active_ruleset_->start();
}

void Game::draw_board(uint32_t *board) {
    active_ruleset_->get_pixels(board);
}

void Game::print_controls() {
    std::cout << "F     : Toggle CUDA processing" << std::endl;
    std::cout << "P     : Print current ruleset info" << std::endl;
    active_ruleset_->print_controls();
}

void Game::tick(void) {
    if(last_ruleset_ != current_ruleset_) {
        change_ruleset(current_ruleset_);
    }
    last_ruleset_ = current_ruleset_;

    active_ruleset_->tick();

    return;
}

