#ifndef _GAME_H
#define _GAME_H

#include <SDL.h>

#include "initializer.h"
#include "rulesets/ruleset.h"


class Game {
    private:
        Ruleset *ruleset_;
        SDL_Window *window_;
        int width_;
        int height_;

        void handle_input(SDL_Event event, bool control, bool shift);

    public:
        Game();
        ~Game();
        int main(void);
};

#endif //_GAME_H
