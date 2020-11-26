
#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <SDL.h>
#include <SDL_image.h>
#include <sstream>
#include <time.h>
#include <thread>
#include <vector>

#include "game.h"
#include "input_manager.h"
#include "movie.h"

#define WIDTH 1080
#define HEIGHT 1080

int main(int argc, char * arg[])
{
    Game game(WIDTH, HEIGHT);
    SDL_Event event;
    MovieWriter *writer = NULL;
    std::vector<uint8_t> writer_pixels(4 * WIDTH * HEIGHT);

    int fps_target = 60;
    bool lock_cursor = false;
    bool running = true, shift = false, control = false;


    static uint8_t data[8] = {0};
    static uint8_t mask[8] = {0};

    srand(time(NULL));

    InputManager::add_var_changer(&fps_target, SDLK_v, 5, 1, INT_MAX, "FPS Target");

    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    SDL_Cursor *cursor = SDL_CreateCursor(data, mask, 8, 8, 0, 0);

    SDL_Window *window = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               WIDTH,                 // width
                               HEIGHT,                // height
                               SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED);
    SDL_SetCursor(cursor);

    while(running) {
        auto start_time = std::chrono::steady_clock::now();

        game.draw_board((uint32_t*)(SDL_GetWindowSurface(window)->pixels));
        SDL_UpdateWindowSurface(window);

        if(writer) {
            game.draw_board((uint32_t*)(&writer_pixels[0]));
            writer->addFrame(&writer_pixels[0]);
        }

        game.tick();

        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                exit(0);
            }
            else if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
                switch(event.key.keysym.sym) {
                    case SDLK_LCTRL:
                    case SDLK_RCTRL:
                        control = event.type == SDL_KEYDOWN;
                        break;
                    case SDLK_LSHIFT:
                    case SDLK_RSHIFT:
                        shift = event.type == SDL_KEYDOWN;
                        break;
                }
            }
            if(event.type == SDL_KEYDOWN) {
                switch(event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        running = false;
                        break;
                    case SDLK_LEFTBRACKET: {
                        //Get the time and convert it to a string.png
                        std::time_t t = std::time(nullptr);
                        std::tm tm = *std::localtime(&t);
                        std::ostringstream oss;
                        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S.png");
                        std::string str = oss.str();

                        IMG_SavePNG(SDL_GetWindowSurface(window), str.c_str());
                        break;
                    }
                    case SDLK_RIGHTBRACKET:
                        if(writer) {
                            std::cout << "Writing video file." << std::endl;
                            delete writer;
                            writer = NULL;
                        }
                        else {
                            std::time_t t = std::time(nullptr);
                            std::tm tm = *std::localtime(&t);
                            std::ostringstream oss;
                            oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
                            std::string str = oss.str();
                            writer = new MovieWriter(str, WIDTH, HEIGHT);
                        }
                        break;
                    case SDLK_l:
                        lock_cursor = !lock_cursor;
                        break;
                }
            }
            game.handle_input(event, control, shift);
            InputManager::handle_input(event, control, shift);
        }
        if(lock_cursor) {
            SDL_WarpMouseInWindow(window, WIDTH/2, HEIGHT/2);
        }

        std::chrono::microseconds frame_delay(1000000/fps_target);
        auto next_frame_time = start_time + frame_delay;
        auto delay_time = next_frame_time - std::chrono::steady_clock::now();
        std::this_thread::sleep_for(delay_time);
    }

    SDL_FreeCursor(cursor);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
