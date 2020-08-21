
#include "input_manager.h"

#include <iostream>


std::set<VarChangeEntry*, decltype(InputManager::compare_entries)*> 
    InputManager::int_changes_ = 
        std::set<VarChangeEntry*, decltype(InputManager::compare_entries)*>
            (InputManager::compare_entries);

std::set<SDL_Keycode> InputManager::used_keys_ = std::set<SDL_Keycode>();

void InputManager::add_var_changer(int *variable, SDL_Keycode key, int multiplier,
                                   int min_value, int max_value, std::string name) {
    if(!used_keys_.insert(key).second) {
        std::cout << "Warning: Attempted rebind on "
                  << SDL_GetKeyName(key) 
                  << " for function " << name 
                  << " not registered." << std::endl;
        return;
    }

    VarChangeEntry *entry = new VarChangeEntry;
    entry->key = key;
    entry->key_pressed = false;
    entry->max_value = max_value;
    entry->min_value = min_value;
    entry->multiplier = multiplier;
    entry->name = name;
    entry->variable = variable;
    int_changes_.insert(entry);
}

bool InputManager::compare_entries(VarChangeEntry *a, VarChangeEntry *b) {
    return (a->key) < (b->key);
}

void InputManager::handle_input(SDL_Event event, bool control, bool shift) {
    int override_value = 0;
    int modify_value = 0;
    int mul = (control ? 2 : 1) * (shift ? 5 : 1);

    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        for(VarChangeEntry *entry: int_changes_) {
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
            override_value = event.key.keysym.sym - SDLK_KP_1 + 1;
        }
        else if(event.key.keysym.sym == SDLK_KP_PLUS) {
            modify_value = 1;
        }
        else if(event.key.keysym.sym == SDLK_KP_MINUS) {
            modify_value = -1;
        }
    }
    if(modify_value || override_value) {
        for(VarChangeEntry *entry: int_changes_) {
            if(entry->key_pressed) {
                if(override_value) {
                    *(entry->variable) = (override_value % 10) * 
                                         entry->multiplier * mul;
                }
                if(modify_value) {
                    *(entry->variable) += modify_value * mul;
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
