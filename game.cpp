
#include <iostream>

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

    for(int i = 0; i < 10; i++) {
        frame_times_.push_back(std::chrono::high_resolution_clock::now());
    }

    ADD_FUNCTION_CALLER(&Game::print_fps, SDL_SCANCODE_X, false, false,
                        "(Game) Print Frames Per Second");

    InputManager::add_int_changer(&current_ruleset_, SDL_SCANCODE_Z, false, false,
                                  0, NUM_RULESETS-1, "(Game) Ruleset");
}

Game::~Game() {
    active_ruleset_->stop();
    for(Ruleset *ruleset: rulesets_) {
        delete ruleset;
    }
    InputManager::remove_var_changer(SDL_SCANCODE_Z, false, false);
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

void Game::print_fps(void) {
    auto total_time = frame_times_.back() - frame_times_.front();
    auto avg_frame_time = total_time / frame_times_.size();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(avg_frame_time).count();
    std::cout << "FPS: " << 1000000.0 / time << std::endl;
}

void Game::tick(void) {
    if(last_ruleset_ != current_ruleset_) {
        change_ruleset(current_ruleset_);
    }
    last_ruleset_ = current_ruleset_;

    frame_times_.pop_front();
    frame_times_.push_back(std::chrono::high_resolution_clock::now());

    active_ruleset_->tick();

    return;
}

