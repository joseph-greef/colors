#ifndef _GAME_H
#define _GAME_H

#include "board.h"
#include "initializer.h"
#include "rulesets/ruleset.h"
#include "screen.h"


class Game {
    private:
        Board _board;
        Initializer _initer;
        Screen _screen;
        Ruleset *_ruleset;

    public:
        Game();
        ~Game();
        int main(void);
};

#endif //_GAME_H
