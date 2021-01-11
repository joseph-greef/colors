
#include <iostream>
#include <random>
#include <tclap/CmdLine.h>

#include "game.h"

int main(int argc, char * argv[])
{
    int width, height, fps_target, square;
    try {
        TCLAP::CmdLine cmd("See README.md/readme.pdf or press ' (single quote) while running for simulation controls.", ' ', "0.1");
        TCLAP::ValueArg<int> fps_arg("f", "fps", "Set the initial FPS target of the simulation, defaults to monitor refresh rate", false, -1, "int");
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

    if(square > 0) {
        width = height = square;
    }

    srand(time(NULL));
    Game game(fps_target, width, height);
    game.main();

    return 0;
}
