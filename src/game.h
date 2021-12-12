#ifndef _GAME_H
#define _GAME_H

#define SDL_MAIN_HANDLED

#include <chrono>
#include <deque>
#include "SDL2/SDL.h"
#include "SDL2/SDL_image.h"

#include "ruleset.h"

struct TempRuleEntry {
    int ruleset_num;
    std::string rule_string;
};

class Game {
    private:
        Ruleset *active_ruleset_;
        int current_ruleset_;
        int fps_target_;
        std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>> frame_times_;
        std::vector<Ruleset*> rulesets_;
        bool running_;
        TempRuleEntry saved_rules_[10];
        SDL_Window *window_;
        Board<Pixel<uint8_t>> *pixels_;
        int width_;
        int height_;

        int change_ruleset(int new_ruleset, int modifier, bool transfer_board);
        void load_rule_string_from_clipboard(void);
        void load_rule_string_from_file(void);
        void load_rule_string_from_temp(int index);
        void print_fps(void);
        void print_rules(void);
        void save_rule_string_to_clipboard(void);
        void save_rule_string_to_file(void);
        void save_rule_string_to_temp(int index);
        void take_screenshot(void);

    public:
        Game(int fps_target, int width, int height);
        ~Game();
        void main(void);
};

#endif //_GAME_H

