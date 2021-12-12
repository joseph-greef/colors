
#include <ctime>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>

#include "clip/clip.h"
#include "game.h"
#include "input_manager.h"

#define FRAMES_TO_AVERAGE 10


Game::Game(int fps_target, int width, int height)
    : active_ruleset_(NULL)
    , current_ruleset_(0)
    , fps_target_(fps_target)
    , running_(true)
{
    SDL_DisplayMode display_mode;

    TempRuleEntry initial_temp_rules[10] = {
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {1, "260 30 2 2 5 1"},
        {3, ""},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
        {0, "0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 4"},
    };

    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    SDL_GetCurrentDisplayMode(0, &display_mode);
    if(height < 1) {
        height = display_mode.h;
    }
    if(width < 1) {
        width = display_mode.w;
    }
    if(fps_target_ < 1) {
        fps_target_ = display_mode.refresh_rate;
    }

    window_ = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               width,                 // width
                               height,                // height
                               SDL_WINDOW_BORDERLESS);
    SDL_ShowCursor(0);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    pixels_ = new Buffer<Pixel<uint8_t>>(width, height,
            static_cast<Pixel<uint8_t>*>(SDL_GetWindowSurface(window_)->pixels));

    rulesets_.push_back(new LifeLike(width, height));
    rulesets_.push_back(new Hodge(width, height));
    rulesets_.push_back(new Ants(width, height));
    active_ruleset_ = rulesets_[current_ruleset_];
    active_ruleset_->start();

    for(int i = 0; i < FRAMES_TO_AVERAGE; i++) {
        frame_times_.push_back(std::chrono::high_resolution_clock::now());
    }

    for(int i = 0; i < 10; i++) {
        ADD_FUNCTION_CALLER_W_ARGS(&Game::load_rule_string_from_temp, VoidFunc,
                   static_cast<SDL_Scancode>(static_cast<int>(SDL_SCANCODE_1) + i),
                   false, false, "Game", "Load rules from temp storage", i);
        ADD_FUNCTION_CALLER_W_ARGS(&Game::save_rule_string_to_temp, VoidFunc,
                   static_cast<SDL_Scancode>(static_cast<int>(SDL_SCANCODE_1) + i),
                   true, false, "Game", "Save rules to temp storage", i);
        saved_rules_[i].ruleset_num = initial_temp_rules[i].ruleset_num;
        saved_rules_[i].rule_string = initial_temp_rules[i].rule_string;
    }

    ADD_FUNCTION_CALLER(&Game::print_rules, SDL_SCANCODE_P, false, false,
                        "Game", "Print current game's rules");
    ADD_FUNCTION_CALLER(&Game::save_rule_string_to_clipboard, SDL_SCANCODE_P, true, false,
                        "Game", "Save rule to clipboard");
    ADD_FUNCTION_CALLER(&Game::save_rule_string_to_file, SDL_SCANCODE_P, false, true,
                        "Game", "Save rule to file");
    ADD_FUNCTION_CALLER(&Game::load_rule_string_from_clipboard, SDL_SCANCODE_R, true, false,
                        "Game", "Load rule from clipboard");
    ADD_FUNCTION_CALLER(&Game::load_rule_string_from_file, SDL_SCANCODE_R, false, true,
                        "Game", "Load rule from file");
    ADD_FUNCTION_CALLER(&Game::print_fps, SDL_SCANCODE_X, false, false,
                        "Game", "Print frames per second");
    ADD_FUNCTION_CALLER(&Game::take_screenshot, SDL_SCANCODE_LEFTBRACKET, false, false,
                        "Game", "Take screenshot");

    ADD_FUNCTION_CALLER_W_ARGS(&Game::change_ruleset, IntFunc, SDL_SCANCODE_Z,
                               false, false, "Game", "Change ruleset", _1, _2, false);
    ADD_FUNCTION_CALLER_W_ARGS(&Game::change_ruleset, IntFunc, SDL_SCANCODE_Z,
                               true, false, "Game", "Change ruleset (transfer buffer)",
                               _1, _2, true);

    InputManager::add_bool_toggler(&running_, SDL_SCANCODE_ESCAPE, false, false,
                                   "Game", "Quit application");

    InputManager::add_input(InputManager::print_controls, SDL_SCANCODE_APOSTROPHE,
                            false, false, "Game", "Print help message");

    InputManager::add_int_changer(&fps_target_, SDL_SCANCODE_V, false, false,
                                  10, INT_MAX, "Game", "Set FPS target");
}

Game::~Game() {
    active_ruleset_->stop();
    for(Ruleset *ruleset: rulesets_) {
        delete ruleset;
    }
    InputManager::remove_var_changer(SDL_SCANCODE_P, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_P, true, false);
    InputManager::remove_var_changer(SDL_SCANCODE_P, false, true);
    InputManager::remove_var_changer(SDL_SCANCODE_R, true, false);
    InputManager::remove_var_changer(SDL_SCANCODE_R, false, true);
    InputManager::remove_var_changer(SDL_SCANCODE_X, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_LEFTBRACKET, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_Z, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Z, true, false);

    InputManager::remove_var_changer(SDL_SCANCODE_ESCAPE, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_APOSTROPHE, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_V, false, false);

    SDL_DestroyWindow(window_);
    SDL_Quit();
}

int Game::change_ruleset(int new_ruleset, int modifier, bool transfer_buffer) {
    if(new_ruleset == INT_MIN) {
        new_ruleset = current_ruleset_;
    }
    new_ruleset += modifier;
    if(new_ruleset != current_ruleset_ &&
       new_ruleset >= 0 &&
       new_ruleset < rulesets_.size())
    {
        Ruleset *old_ruleset = active_ruleset_;

        active_ruleset_->stop();
        InputManager::trigger_reset();
        active_ruleset_ = rulesets_[new_ruleset];
        active_ruleset_->start();
        current_ruleset_ = new_ruleset;

        if(transfer_buffer) {
            std::size_t selected_type = active_ruleset_->select_buffer_type(
                    old_ruleset->buffer_types_provided());
            if(selected_type == RGBA_BUFFER) {
                old_ruleset->get_pixels(pixels_);
                active_ruleset_->set_buffer(pixels_, RGBA_BUFFER);
            }
            else if(selected_type != NOT_COMPATIBLE) {
                active_ruleset_->set_buffer(old_ruleset->get_buffer(selected_type),
                                           selected_type);
            }
        }
    }
    return 0;
}

void Game::load_rule_string_from_clipboard(void) {
    std::string clipboard;
    clip::get_text(clipboard);
    active_ruleset_->load_rule_string(clipboard);
}

void Game::load_rule_string_from_file(void) {
    std::string line;
    std::ostringstream name_ss;
    std::ifstream rules_file;
    std::vector<std::string> rules_lines;
    std::vector<std::string> out_lines;

    name_ss << active_ruleset_->get_name() << ".txt";
    rules_file.open(name_ss.str(), std::ios::in);

    while(std::getline(rules_file, line)) {
        rules_lines.push_back(line);
    }

    std::sample(rules_lines.begin(), rules_lines.end(), std::back_inserter(out_lines),
                1, std::mt19937{std::random_device{}()});

    std::cout << out_lines[0] << std::endl;

    active_ruleset_->load_rule_string(out_lines[0]);
}

void Game::load_rule_string_from_temp(int index) {
    change_ruleset(saved_rules_[index].ruleset_num, 0, true);
    active_ruleset_->load_rule_string(saved_rules_[index].rule_string);
    std::cout << std::endl << index << " | " << saved_rules_[index].rule_string << std::endl;
}

void Game::main(void) {
    SDL_Event event;

    while(running_) {
        auto start_time = std::chrono::high_resolution_clock::now();

        active_ruleset_->get_pixels(pixels_);
        SDL_UpdateWindowSurface(window_);

        active_ruleset_->tick();

        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running_ = false;
            }
            else {
                InputManager::handle_input(event);
            }
        }

        frame_times_.pop_front();
        frame_times_.push_back(std::chrono::high_resolution_clock::now());

        std::chrono::microseconds frame_delay(1000000/fps_target_);
        auto next_frame_time = start_time + frame_delay;
        auto delay_time = next_frame_time - std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(delay_time);
    }
}

void Game::print_fps(void) {
    auto total_time = frame_times_.back() - frame_times_.front();
    auto avg_frame_time = total_time / (frame_times_.size() - 1);
    uint64_t time = std::chrono::duration_cast<std::chrono::microseconds>(avg_frame_time).count();
    std::cout << "FPS: " << 1000000.0 / time << std::endl;
}

void Game::print_rules(void) {
    active_ruleset_->print_human_readable_rules();
}

void Game::save_rule_string_to_clipboard(void) {
    std::string rules = active_ruleset_->get_rule_string();
    std::cout << rules << std::endl;
    clip::set_text(rules);
}

void Game::save_rule_string_to_file(void) {
    std::ofstream rules_file;
    std::ostringstream name_ss;
    name_ss << active_ruleset_->get_name() << ".txt";
    rules_file.open(name_ss.str(), std::ios::out | std::ios::app);
    rules_file << active_ruleset_->get_rule_string() << "\n";
    rules_file.close();
}

void Game::save_rule_string_to_temp(int index) {
    saved_rules_[index].ruleset_num = current_ruleset_;
    saved_rules_[index].rule_string = active_ruleset_->get_rule_string();
    std::cout << std::endl << index << " | " <<
                 saved_rules_[index].rule_string << std::endl;
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


