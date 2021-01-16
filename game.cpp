
#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "game.h"
#include "input_manager.h"


Game::Game(int fps_target, int width, int height)
    : active_ruleset_(NULL)
    , current_ruleset_(0)
    , fps_target_(fps_target)
    , last_ruleset_(current_ruleset_)
    , running_(true)
    , width_(width)
    , height_(height)
{
    SDL_DisplayMode display_mode;

    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    SDL_GetCurrentDisplayMode(0, &display_mode);
    if(height_ < 1) {
        height_ = display_mode.h;
    }
    if(width_ < 1) {
        width_ = display_mode.w;
    }
    if(fps_target_ < 1) {
        fps_target_ = display_mode.refresh_rate;
    }

    window_ = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               width_,                 // width
                               height_,                // height
                               SDL_WINDOW_BORDERLESS);
    SDL_ShowCursor(0);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    rulesets_.push_back(new LifeLike(width_, height_));
    rulesets_.push_back(new Hodge(width_, height_));
    rulesets_.push_back(new Ants(width_, height_));
    active_ruleset_ = rulesets_[current_ruleset_];
    active_ruleset_->start();

    for(int i = 0; i < 10; i++) {
        frame_times_.push_back(std::chrono::high_resolution_clock::now());
    }

    ADD_FUNCTION_CALLER(&Game::print_fps, SDL_SCANCODE_X, false, false,
                        "Game", "Print frames per second");
    ADD_FUNCTION_CALLER(&Game::print_rules, SDL_SCANCODE_P, false, true,
                        "Game", "Print current game's ruleset");
    ADD_FUNCTION_CALLER(&Game::take_screenshot, SDL_SCANCODE_LEFTBRACKET, false, false,
                        "Game", "Take screenshot");

    InputManager::add_bool_toggler(&running_, SDL_SCANCODE_ESCAPE, false, false,
                                   "Game", "Quit application");

    InputManager::add_input(InputManager::print_controls, SDL_SCANCODE_APOSTROPHE,
                            false, false, "Game", "Print help message");

    InputManager::add_int_changer(&fps_target_, SDL_SCANCODE_V, false, false,
                                  10, INT_MAX, "Game", "Set FPS target");
    InputManager::add_int_changer(&current_ruleset_, SDL_SCANCODE_Z, false, false,
                                  0, NUM_RULESETS-1, "Game", "Change ruleset");
}

Game::~Game() {
    active_ruleset_->stop();
    for(Ruleset *ruleset: rulesets_) {
        delete ruleset;
    }
    InputManager::remove_var_changer(SDL_SCANCODE_X, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_LEFTBRACKET, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_ESCAPE, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_APOSTROPHE, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_V, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Z, false, false);

    SDL_DestroyWindow(window_);
    SDL_Quit();
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

void Game::main(void) {
    SDL_Event event;

    while(running_) {
        auto start_time = std::chrono::high_resolution_clock::now();

        active_ruleset_->get_pixels(
                static_cast<uint32_t*>(SDL_GetWindowSurface(window_)->pixels));
        SDL_UpdateWindowSurface(window_);
        tick();

        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running_ = false;
            }
            else {
                InputManager::handle_input(event);
            }
        }

        std::chrono::microseconds frame_delay(1000000/fps_target_);
        auto next_frame_time = start_time + frame_delay;
        auto delay_time = next_frame_time - std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(delay_time);
    }
}

void Game::print_fps(void) {
    auto total_time = frame_times_.back() - frame_times_.front();
    auto avg_frame_time = total_time / frame_times_.size();
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(avg_frame_time).count();
    std::cout << "FPS: " << 1000000.0 / time << std::endl;
}

void Game::print_rules(void) {
    active_ruleset_->print_rules();
}

void Game::take_screenshot(void) {
    //Get the time and convert it to a string.png
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.png");
    std::string str = oss.str();

    IMG_SavePNG(SDL_GetWindowSurface(window_), str.c_str());
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

