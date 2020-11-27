
#include "input_manager.h"

#include <iostream>


std::set<VarChangeEntry*, decltype(InputManager::compare_entries)*> 
    InputManager::int_changes_ = 
        std::set<VarChangeEntry*, decltype(InputManager::compare_entries)*>
            (InputManager::compare_entries);

std::set<SDL_Keycode> InputManager::used_keys_ = std::set<SDL_Keycode>();

std::vector<VarChangeEntry*> InputManager::left_mouse_vars_ = std::vector<VarChangeEntry*>();

std::vector<VarChangeEntry*> InputManager::right_mouse_vars_ = std::vector<VarChangeEntry*>();

void InputManager::add_var_changer(int *variable, SDL_Keycode key,
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
    entry->override_value = 0;
    entry->overridden = false;
    entry->name = name;
    entry->variable = variable;
    int_changes_.insert(entry);
}

bool InputManager::compare_entries(VarChangeEntry *a, VarChangeEntry *b) {
    return (a->key) < (b->key);
}

void InputManager::handle_input(SDL_Event event, bool control, bool shift) {
    int override_value = -1;
    int modify_value = 0;
    int mul = (control ? 2 : 1) * (shift ? 5 : 1);

    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        for(VarChangeEntry *entry: int_changes_) {
            if(event.key.keysym.sym == entry->key) {
                entry->key_pressed = (event.type == SDL_KEYDOWN);
                if(event.type == SDL_KEYUP && entry->overridden) {
                    modify_entry(entry, entry->override_value, 0);
                }
                entry->override_value = 0;
                entry->overridden = false;
                break;
            }
        }
    }

    if(event.type == SDL_MOUSEMOTION) {
        if(event.motion.state & SDL_BUTTON_LMASK) {
            for(VarChangeEntry *entry: left_mouse_vars_) {
                modify_entry(entry, -1, event.motion.xrel);
            }
        }
        if(event.motion.state & SDL_BUTTON_RMASK) {
            for(VarChangeEntry *entry: right_mouse_vars_) {
                modify_entry(entry, -1, event.motion.yrel);
            }
        }

    }
    else if(event.type == SDL_MOUSEBUTTONDOWN) {
        if(control && (event.button.button == SDL_BUTTON_LEFT ||
                       event.button.button == SDL_BUTTON_RIGHT )) {
            std::vector<VarChangeEntry*> *active;
            if(event.button.button == SDL_BUTTON_LEFT) {
                active = &left_mouse_vars_;
            }
            else {
                active = &right_mouse_vars_;
            }
            active->clear();
            for(VarChangeEntry *entry: int_changes_) {
                if(entry->key_pressed) {
                    active->push_back(entry);
                }
            }
        }

    }
    else if(event.type == SDL_KEYDOWN) {
        if(event.key.keysym.sym >= SDLK_KP_1 && 
           event.key.keysym.sym <= SDLK_KP_0) {
            //order is KP_1, KP_2 ... KP_0
            override_value = event.key.keysym.sym - SDLK_KP_1 + 1;
            override_value = (override_value % 10);
        }
        else if(event.key.keysym.sym == SDLK_KP_PLUS) {
            modify_value = mul;
        }
        else if(event.key.keysym.sym == SDLK_KP_MINUS) {
            modify_value = -mul;
        }
    }
    else if(event.type == SDL_MOUSEWHEEL) {
        modify_value = event.wheel.y * mul;
    }
    if(modify_value) {
        for(VarChangeEntry *entry: int_changes_) {
            if(entry->key_pressed) {
                modify_entry(entry, -1, modify_value);
            }
        }
    }
    if(override_value != -1) {
        for(VarChangeEntry *entry: int_changes_) {
            if(entry->key_pressed) {
                entry->override_value *= 10;
                entry->override_value += override_value;
                entry->overridden = true;
            }
        }
    }
}

void InputManager::modify_entry(VarChangeEntry *entry, int override_value, int modify_value) {
    if(override_value != -1) {
        *(entry->variable) = override_value;
    }
    if(modify_value) {
        *(entry->variable) += modify_value;
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


void InputManager::remove_var_changer(SDL_Keycode key) {
    VarChangeEntry *toErase = NULL;

    if(used_keys_.erase(key) == 0) {
        std::cout << "Attempted to remove key "
                  << SDL_GetKeyName(key) 
                  << " which wasn't added." << std::endl;
    }
    for(VarChangeEntry *entry: int_changes_) {
        if(key == entry->key) {
            toErase = entry;
            break;
        }
    }
    if(toErase) {
        int_changes_.erase(toErase);
        delete(toErase);
    }
    else {
        std::cout << "Didn't find "
                  << SDL_GetKeyName(key) 
                  << " in int_changes_" << std::endl;
    }
}

void InputManager::reset() {
    left_mouse_vars_.clear();
    right_mouse_vars_.clear();
}
