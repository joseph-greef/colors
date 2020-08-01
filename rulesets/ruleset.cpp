
#include <cmath>

#include "ruleset.h"
#include "rainbows.h"

Ruleset::Ruleset(int width, int height)
    : height_(height)
    , width_(width)
{
}

Ruleset::~Ruleset() {
}

void Ruleset::add_var_changer(int *variable, SDL_Keycode key, int multiplier) {
    VarChangeEntry *entry = new VarChangeEntry;
    entry->variable = variable;
    entry->key = key;
    entry->multiplier = multiplier;

    var_changes_.push_back(entry);
}

void Ruleset::handle_var_changers(SDL_Event event, bool control, bool shift) {
    int *variable = NULL;
    int multiplier = 0;
    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        for(VarChangeEntry *entry: var_changes_) {
            if(event.key.keysym.sym == entry->key) {
                entry->key_pressed = (event.type == SDL_KEYDOWN);
                break;
            }
        }
    }

    if(variable != NULL) {
        
    } 
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
                    check_x = (i + width_) % width_;
                    check_y = (j + height_) % height_;
                    //and check the coordinate, if it's alive increase count
                    if(board[check_y*width_+check_x] > 0)
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


                check_x = (i + width_) % width_;
                check_y = (j + height_) % height_;
                //and check the coordinate, if it's alive increase count
                if(board[check_y*width_+check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}

