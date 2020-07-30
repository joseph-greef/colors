//
//  main.m
//  Colors
//
//
#include <iostream>
#include <time.h>
#include "board.h"
#include "screen.h"
#include "initializer.h"
#include "RuleGenerator.h"
#include "kernel.cuh"
#include <unistd.h>

#define timeit 1

static int frame_delay = 0;


int main(int argc, char * arg[])
{
    
    clock_t t;
    SDL_Event event;

    srand (time(NULL));

    Board board;
    Screen screen(&board);
    Initializer initer(&board);

    board.set_update_algorithm(0);
    initer.init_board();

    uint8_t color_speed_divisor;
    

    bool running = true;
    while(running) {
        //translate board to pixels
        screen.draw_board();
        //and draw it
        screen.update_window();
        board.update_board();


        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && 
                                          event.key.keysym.sym == SDLK_ESCAPE)) {
                exit(0);
            }
        }
    }

    //return (this deconstructs screen and board and thus cleans up everything)
    return 0;
}
