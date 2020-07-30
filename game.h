#ifndef _GAME_H
#define _GAME_H

#include "board.h"
#include "initializer.h"
#include "screen.h"


class Game {
    private:
        Board _board;
        Initializer _initer;
        Screen _screen;

    public:
        Game();
        ~Game();
        int main(void);
};

#endif //_GAME_H
