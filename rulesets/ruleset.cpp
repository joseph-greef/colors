
#include <cmath>

#include "ruleset.h"
#include "rainbows.h"

Ruleset::Ruleset(int width, int height)
    : _height(height)
    , _width(width)
{
}

int Ruleset::get_num_alive_neighbors(int *board, int x, int y,
                                            int radius,
                                            NeighborhoodType type) {
    int check_x = 0;
    int check_y = 0; 
    int count = 0;

    if(type == VonNeuman) {
        for(int i = x - radius; i <= x + radius; i++) {
            for(int j = y - radius; j <= y + radius; j++) {
                if(j==y && i==x)
                    continue;
                if(abs(i-x)+abs(j-y) <= radius) {


                    check_x = (i + _width) % _width;
                    check_y = (j + _height) % _height;
                    //and check the coordinate, if it's alive increase count
                    if(board[check_y*_width+check_x] > 0)
                        count++;
                }
            }
        }
    }
    else {
        for(int i = x - radius; i <= x + radius; i++) {
            for(int j = y - radius; j <= y + radius; j++) {
                if(j==y && i==x)
                    continue;


                check_x = (i + _width) % _width;
                check_y = (j + _height) % _height;
                //and check the coordinate, if it's alive increase count
                if(board[check_y*_width+check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}

