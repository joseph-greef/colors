#ifndef _GAME_H
#define _GAME_H

#include <SDL.h>

#include "initializer.h"
#include "rulesets/ruleset.h"


class Game {
    private:
        int current_ruleset_;
        bool lock_cursor_;
        Ruleset *ruleset_;
        SDL_Window *window_;
        const int width_;
        const int height_;

        void change_ruleset(int new_ruleset);
        void handle_input(SDL_Event event, bool control, bool shift);

    public:
        Game();
        ~Game();
        int main(void);
};

#endif //_GAME_H
