#ifndef _GAME_H
#define _GAME_H

#define SDL_MAIN_HANDLED

#include <chrono>
#include <deque>
#include "SDL2/SDL.h"
#include "SDL2/SDL_image.h"

#include "ruleset.h"

class Game {
    private:
        Ruleset *active_ruleset_;
        int current_ruleset_;
        int fps_target_;
        int last_ruleset_;
        std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>> frame_times_;
        std::vector<Ruleset*> rulesets_;
        bool running_;
        SDL_Window *window_;
        int width_;
        int height_;

        void change_ruleset(int new_ruleset);
        void print_fps(void);
        void print_rules(void);
        void take_screenshot(void);

    public:
        Game(int fps_target, int width, int height);
        ~Game();
        void draw_board(uint32_t *board);
        void main(void);
        void tick(void);
};

#endif //_GAME_H

