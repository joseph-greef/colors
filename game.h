#ifndef _GAME_H
#define _GAME_H

#include "board.h"
#include "initializer.h"
#include "rulesets/ruleset.h"
#include "screen.h"


class Game {
    private:
        //Board _board;
        //Initializer _initer;
        Ruleset *_ruleset;
        //Screen _screen;
        SDL_Window *_window;
        int _width;
        int _height;

    public:
        Game();
        ~Game();
        int main(void);
};

#endif //_GAME_H
