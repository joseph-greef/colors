
#include <climits>
#include <cmath>
#include <iostream>

#include "ruleset.h"
#include "rainbows.h"

Ruleset::Ruleset(int width, int height)
    : height_(height)
    , width_(width)
{
}

Ruleset::~Ruleset() {
}


void Ruleset::add_var_changer(int *variable, SDL_Keycode key, int multiplier,
                              const char *name) {
    add_var_changer(variable, key, multiplier, INT_MIN, INT_MAX, name);
}
void Ruleset::add_var_changer(int *variable, SDL_Keycode key, int multiplier,
                              int min_value, int max_value, const char *name) {
    VarChangeEntry *entry = new VarChangeEntry;
    entry->key = key;
    entry->key_pressed = false;
    entry->max_value = max_value;
    entry->min_value = min_value;
    entry->multiplier = multiplier;
    entry->name = name;
    entry->variable = variable;

    var_changes_.push_back(entry);
}

void Ruleset::handle_var_changers(SDL_Event event, bool control, bool shift) {
    int base = 0;
    int modifier = 0;
    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        for(VarChangeEntry *entry: var_changes_) {
            if(event.key.keysym.sym == entry->key) {
                entry->key_pressed = (event.type == SDL_KEYDOWN);
                break;
            }
        }
    }

    if(event.type == SDL_KEYDOWN) {
        if(event.key.keysym.sym >= SDLK_KP_1 && 
           event.key.keysym.sym <= SDLK_KP_0) {
            //order is KP_1, KP_2 ... KP_0
            base = event.key.keysym.sym - SDLK_KP_1 + 1;
        }
        else if(event.key.keysym.sym == SDLK_KP_PLUS) {
            modifier = 1;
        }
        else if(event.key.keysym.sym == SDLK_KP_MINUS) {
            modifier = -1;
        }
    }
    if(modifier || base) {
        for(VarChangeEntry *entry: var_changes_) {
            if(entry->key_pressed) {
                int mul = (control ? 2 : 1) * (shift ? 5 : 1);
                if(base) {
                    *(entry->variable) = (base % 10) * entry->multiplier * mul;
                }
                if(modifier) {
                    *(entry->variable) += modifier * mul;
                }
                
                if(*(entry->variable) < entry->min_value) {
                    *(entry->variable) = entry->min_value;
                }
                if(*(entry->variable) > entry->max_value) {
                    *(entry->variable) = entry->max_value;
                }
                
                std::cout << entry->name 
                          << ": " 
                          << *(entry->variable) 
                          << std::endl;
            }
        }
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

