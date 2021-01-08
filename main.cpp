#define SDL_MAIN_HANDLED

#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "SDL2/SDL.h"
#include "SDL2/SDL_image.h"
#include <sstream>
#include <tclap/CmdLine.h>
#include <thread>
#include <vector>

#include "game.h"
#include "input_manager.h"

int main(int argc, char * argv[])
{
    int width, height, fps_target, square;
    try {
        TCLAP::CmdLine cmd("See README.md/readme.pdf or press ' (single quote) while running for simulation controls.", ' ', "0.1");
        TCLAP::ValueArg<int> fps_arg("f", "fps", "Set the initial FPS target of the simulation, defaults to 144Hz", false, 144, "int");
        cmd.add(fps_arg);
        TCLAP::ValueArg<int> square_arg("s", "square", "If set positive, overrides other size controls to create a square of the given size", false, -1, "int");
        cmd.add(square_arg);
        TCLAP::ValueArg<int> width_arg("c", "cols", "Set the width of sim, uses screen width if unset or invalid", false, -1, "int");
        cmd.add(width_arg);
        TCLAP::ValueArg<int> height_arg("r", "rows", "Set the height of sim, uses screen height if unset or invalid", false, -1, "int");
        cmd.add(height_arg);

        cmd.parse(argc, argv);

        width = width_arg.getValue();
        height = height_arg.getValue();
        fps_target = fps_arg.getValue();
        square = square_arg.getValue();
    } catch (TCLAP::ArgException &e) {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cout << "ERROR SDL_Init" << std::endl;
        exit(1);
    }

    SDL_DisplayMode display_mode;
    SDL_GetCurrentDisplayMode(0, &display_mode);

    if(square > 0) {
        width = height = square;
    }
    else {
        if(width < 1) {
            width = display_mode.w;
        }
        if(height < 1) {
            height = display_mode.h;
        }
    }

    srand(time(NULL));
    Game game(width, height);

    //Main loop control/input variables
    bool running = true;
    SDL_Event event;

    std::vector<uint8_t> writer_pixels(4 * width * height);

    InputManager::add_input(InputManager::print_controls, SDL_SCANCODE_APOSTROPHE,
                            false, false, "(Main) Print help message");
    InputManager::add_int_changer(&fps_target, SDL_SCANCODE_V, false, false,
                                  10, INT_MAX, "(Main) Set FPS target");

    SDL_Window *window = SDL_CreateWindow("Colors",               // window title
                               SDL_WINDOWPOS_CENTERED, // x position
                               SDL_WINDOWPOS_CENTERED, // y position
                               width,                 // width
                               height,                // height
                               SDL_WINDOW_BORDERLESS | SDL_WINDOW_MAXIMIZED);
    SDL_ShowCursor(0);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    while(running) {
        auto start_time = std::chrono::high_resolution_clock::now();

        game.draw_board((uint32_t*)(SDL_GetWindowSurface(window)->pixels));
        SDL_UpdateWindowSurface(window);

        game.tick();

        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                exit(0);
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
                }
            }
            InputManager::handle_input(event);
        }

        std::chrono::microseconds frame_delay(1000000/fps_target);
        auto next_frame_time = start_time + frame_delay;
        auto delay_time = next_frame_time - std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(delay_time);
    }

    InputManager::remove_var_changer(SDL_SCANCODE_V, false, false);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
