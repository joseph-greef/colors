//
//  main.m
//  Colors
//
//
#include <iostream>
#include <time.h>
#include <SDL.h>
#include "board.h"
#include "screen.h"
#include "initializer.h"
#include "RuleGenerator.h"
#include "kernel.cuh"



//this function checks if there is user input and reacts accordingly
//returns whether the program should keep running,
//false if it should quit, true if it should continue
bool do_user_input(Board *board, Screen *screen, Initializer *initer) {
    static SDL_Event events;
    while(SDL_PollEvent(&events)) {
        switch(events.type) {
            case SDL_QUIT:
                return false;
                break;
            case SDL_KEYDOWN:
                switch(events.key.keysym.sym) {
                    // this changes the density of live cells that random scenes
                    // are generated with-> The integer is the percent alive cells
                    case SDLK_0:
                        initer->set_density(0);
                        break;
                    case SDLK_1:
                        initer->set_density(10);
                        break;
                    case SDLK_2:
                        initer->set_density(20);
                        break;
                    case SDLK_3:
                        initer->set_density(30);
                        break;
                    case SDLK_4:
                        initer->set_density(40);
                        break;
                    case SDLK_5:
                        initer->set_density(50);
                        break;
                    case SDLK_6:
                        initer->set_density(60);
                        break;
                    case SDLK_7:
                        initer->set_density(70);
                        break;
                    case SDLK_8:
                        initer->set_density(80);
                        break;
                    case SDLK_9:
                        initer->set_density(90);
                        break;
                    case SDLK_LSHIFT:
                        board->toggle_pause();
                        break;
                    
                    //makes a center square of side length density/10
                    case SDLK_a:
                        initer->init_center_dot();
                        break;
                    //set the update algorithm to non-deterministic
                    case SDLK_b:
                        board->set_update_algorithm(3);
                        break;
                    //toggles between black and white and colors
                    case SDLK_c:
                        screen->flip_draw_colors();
                        break;
                    //randomizes the cellular automata ruleset
                    case SDLK_d:
                        board->randomize_rules();
                        screen->reset_colors();
                        break;
                    //initilizes num_gliders/4 dots in each quadrant symmetrically
                    case SDLK_e:
                        initer->init_quadrants();
                        break;
                    //switch between GPU and CPU calculations. F is for fast!
                    case SDLK_f:
                        board->toggle_use_gpu();
                        break;
                    //creates num_gliders gliders in random positions on the board
                    case SDLK_g:
                        initer->init_gliders();
                        break;
                    //set update algorithm to hodge
                    case SDLK_h:
                        board->set_update_algorithm(4);
                        initer->init_hodge_board(board->get_hodge_rules()[3]);
                        break;
                    //changes rules to life
                    case SDLK_i:
                        board->initialize_rules();
                        screen->reset_colors();
                        break;
                    //randomizes smooth rules and makes a new smooth board
                    case SDLK_j:
                        board->randomize_rules_smooth();
                        break;
                    case SDLK_k:
                        initer->init_smooth_life();
                        break;
                    //sets the update algorithm to larger than life
                    case SDLK_l:
                        board->set_update_algorithm(2);
                        break;
                    //set board to smooth automata
                    case SDLK_m:
                        board->set_update_algorithm(1);
                        initer->init_smooth_life();
                        screen->flip_draw_smooth();
                        break;
                    //set board to normal automata
                    case SDLK_n:
                        board->set_update_algorithm(0);
                        screen->flip_draw_smooth();
                        break;
                    //adds a circle to the board
                    case SDLK_o:
                        initer->init_circle();
                        break;
                    case SDLK_p:
                        board->print_rules();
                        break;
                    //quits
                    case SDLK_q:
                        return false;
                        break;
                    //randomizes the cellular automata ruleset with nondeterministic behavior
                    case SDLK_r:
                        board->randomize_rules_non_deterministic();
                        screen->reset_colors();
                        break;
                    //initializes a square shell at the center parallel to the screen
                    case SDLK_s:
                        initer->init_square_shell();
                        break;
                    //initializes a polygon in the center with num_gliders size
                    case SDLK_t:
                        initer->init_polygon_shell();
                        break;
                    //toggles between having the background change colors
                    case SDLK_v:
                        board->toggle_changing_background();
                        break;
                    //initializes a circle shell at the center of the board
                    case SDLK_w:
                        initer->init_circle_shell();
                        break;
                    //randomize colors
                    case SDLK_x:
                        screen->reset_colors();
                        break;
                    case SDLK_y:
                        initer->init_1D_board();
                        break;
                    //clears the board
                    case SDLK_z:
                        initer->clear_board(board->get_board());
                        break;
                    //increases the number of gliders by 1
                    case SDLK_UP:
                        initer->modify_gliders(1);
                        board->modify_num_faders(1);
                        break;
                    //decreases the number of gliders to generate by 1
                    case SDLK_DOWN:
                        initer->modify_gliders(-1);
                        board->modify_num_faders(-1);
                        break;
                    //increases the number of gliders by 4
                    case SDLK_RIGHT:
                        initer->modify_gliders(4);
                        board->modify_num_faders(4);
                        break;
                    //decreases the number of gliders to generate by 4
                    case SDLK_LEFT:
                        initer->modify_gliders(-4);
                        board->modify_num_faders(-4);
                        break;
                    //randomizes the whole board
                    case SDLK_SPACE:
                        initer->init_board();
                       // initer->init_smooth_life();
                        break;
                   
                    case SDLK_COMMA:
                        board->set_update_algorithm(5);
                        initer->init_1D_board();
                        break;


                    case SDLK_EQUALS:
                        board->rules_pretty();
                        screen->reset_colors();
                        break;
                    case SDLK_MINUS:
                        board->rules_not_pretty();
                        screen->reset_colors();
                        break;
                    case SDLK_LEFTBRACKET:
                        board->rules_not_pretty_float();
                        screen->reset_colors();
                        break;
                    case SDLK_RIGHTBRACKET:
                        board->rules_pretty_float();
                        screen->reset_colors();
                        break;

                    // these change how fast the colors cycle, higher divisor means slower
                    case SDLK_F1:
                        screen->set_color_speed_divisor(1);
                        break;
                    case SDLK_F2:
                        screen->set_color_speed_divisor(2);
                        break;
                    case SDLK_F3:
                        screen->set_color_speed_divisor(4);
                        break;
                    case SDLK_F4:
                        screen->set_color_speed_divisor(7);
                        break;
                    case SDLK_F5:
                        screen->set_color_speed_divisor(11);
                        break;
                    case SDLK_F6:
                        screen->set_color_speed_divisor(16);
                        break;
                    case SDLK_F7:
                        screen->set_color_speed_divisor(22);
                        break;
                    case SDLK_F8:
                        screen->set_color_speed_divisor(29);
                        break;
                    case SDLK_F9:
                        screen->set_color_speed_divisor(37);
                        break;
                    case SDLK_F10:
                        screen->set_color_speed_divisor(46);
                        break;
                    case SDLK_F11:
                        screen->set_color_speed_divisor(56);
                        break;
                    case SDLK_F12:
                        screen->set_color_speed_divisor(67);
                        break;

                }
        }
    }
    return true;
}


void teststuff() {
    RuleGenerator gen(20);

    float *seed = (float*) malloc(18*sizeof(float));
    for(int i = 0; i < 18; i++) {
        seed[i] = 1;
    }
    gen.print_array(seed);
    gen.add_seed(seed);
    seed = gen.generate_one_mean_float();
    gen.print_array(seed);
    return;
}

int main(int argc, char * arg[])
{
    srand (time(NULL));

    clock_t t;

    //teststuff();
    //return 0;
    Board board;
    //return 0;
    Screen screen(&board);
    Initializer initer(&board);

    board.set_update_algorithm(0);
    initer.init_board();

    
    bool running = true;
    while(running) {
#if timeit
        printf("%f FPS\n", ((float)CLOCKS_PER_SEC)/(clock() - t));
        t = clock();

#endif //timeit
        //translate board to pixels
        screen.draw_board();
        //and draw it
        screen.update_window();

        board.update_board();

        running = do_user_input(&board, &screen, &initer);

    }

    //return (this deconstructs screen and board and thus cleans up everything)
    return 0;
}
