
#include "game.h"
#include <random>
#include <time.h>

int main(int argc, char * arg[])
{
    Game game;
    srand(time(NULL));
    return game.main();
}
