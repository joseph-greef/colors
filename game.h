#ifndef _GAME_H
#define _GAME_H

#include "rulesets/ruleset.h"

class Game {
    private:
        Ruleset *active_ruleset_;
        int current_ruleset_;
        int last_ruleset_;
        bool lock_cursor_;
        SDL_Cursor *cursor_;
        Ruleset *rulesets_[NUM_RULESETS];
        SDL_Window *window_;
        bool writingVideo_;
        const int width_;
        const int height_;

        void change_ruleset(int new_ruleset);

    public:
        Game(int width, int height);
        ~Game();
        void draw_board(uint32_t *board);
        void handle_input(SDL_Event event, bool control, bool shift);
        void tick(void);
};

#endif //_GAME_H

