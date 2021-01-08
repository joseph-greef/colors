#ifndef _GAME_H
#define _GAME_H

#include <chrono>
#include <deque>

#include "ruleset.h"

class Game {
    private:
        Ruleset *active_ruleset_;
        int current_ruleset_;
        int last_ruleset_;
        bool lock_cursor_;
        SDL_Cursor *cursor_;
        std::deque<std::chrono::time_point<std::chrono::high_resolution_clock>> frame_times_;
        std::vector<Ruleset*> rulesets_;
        SDL_Window *window_;
        const int width_;
        const int height_;

        void change_ruleset(int new_ruleset);
        void print_fps(void);

    public:
        Game(int width, int height);
        ~Game();
        void draw_board(uint32_t *board);
        void tick(void);
};

#endif //_GAME_H

