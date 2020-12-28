
#include "input_manager.h"

#include <iostream>


std::list<BoolTogglerEntry> InputManager::bool_toggles_ =
                                            std::list<BoolTogglerEntry>();
std::list<FunctionCallerEntry> InputManager::function_callers_ =
                                            std::list<FunctionCallerEntry>();
std::list<IntChangeEntry> InputManager::int_changes_ =
                                            std::list<IntChangeEntry>();

std::set<SDL_Keycode> InputManager::used_keys_ = std::set<SDL_Keycode>();

std::vector<IntChangeEntry*> InputManager::left_mouse_vars_ = std::vector<IntChangeEntry*>();

std::vector<IntChangeEntry*> InputManager::right_mouse_vars_ = std::vector<IntChangeEntry*>();

void InputManager::add_bool_toggler(bool *variable, SDL_Keycode key,
                                    std::string name) {
    if(!check_and_insert_key(key, name)) {
        return;
    }

    BoolTogglerEntry entry(key, name, variable);
    bool_toggles_.push_back(entry);
    bool_toggles_.sort();
}

void InputManager::add_function_caller(std::function<void(bool, bool)> function,
                                       SDL_Keycode key, std::string name) {
    if(!check_and_insert_key(key, name)) {
        return;
    }

    FunctionCallerEntry entry(key, name, function);
    function_callers_.push_back(entry);
    function_callers_.sort();
}

void InputManager::add_int_changer(int *variable, SDL_Keycode key,
                                   int min_value, int max_value, std::string name) {
    if(!check_and_insert_key(key, name)) {
        return;
    }

    IntChangeEntry entry(key, name, max_value, min_value, variable);
    int_changes_.push_back(entry);
    int_changes_.sort();
}

bool InputManager::check_and_insert_key(SDL_Keycode key, std::string name) {
    if(!used_keys_.insert(key).second) {
        std::cout << std::endl
                  << "Warning: Attempted rebind on "
                  << SDL_GetKeyName(key) 
                  << " for function " << name 
                  << " not registered." << std::endl;

        std::cout << "Previously registered input is ";
        for(BoolTogglerEntry entry: bool_toggles_) {
            if(entry.key == key) {
                std::cout << entry.name << std::endl;
            }
        }
        for(FunctionCallerEntry entry: function_callers_) {
            if(entry.key == key) {
                std::cout << entry.name << std::endl;
            }
        }
        for(IntChangeEntry entry: int_changes_) {
            if(entry.key == key) {
                std::cout << entry.name << std::endl;
            }
        }
        return false;
    }
    return true;
}

void InputManager::handle_input(SDL_Event event, bool control, bool shift) {
    handle_bool_events(event, control, shift);
    handle_function_events(event, control, shift);
    handle_int_events(event, control, shift);
}

void InputManager::handle_bool_events(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        for(BoolTogglerEntry entry: bool_toggles_) {
            if(event.key.keysym.sym == entry.key) {
                *(entry.variable)= !(*(entry.variable));
            }
        }
    }
}

void InputManager::handle_function_events(SDL_Event event, bool control, bool shift) {
    if(event.type == SDL_KEYDOWN) {
        for(FunctionCallerEntry entry: function_callers_) {
            if(event.key.keysym.sym == entry.key) {
                entry.function(control, shift);
            }
        }
    }
}

void InputManager::handle_int_events(SDL_Event event, bool control, bool shift) {
    int override_value = -1;
    int modify_value = 0;
    int mul = (control ? 2 : 1) * (shift ? 5 : 1);

    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        auto entry = find(int_changes_.begin(), int_changes_.end(), event.key.keysym.sym);
        if(entry != int_changes_.end()) {
            entry->key_pressed = (event.type == SDL_KEYDOWN);
            if(event.type == SDL_KEYUP && entry->overridden) {
                modify_int_entry(&*entry, entry->override_value, 0);
            }
            entry->override_value = 0;
            entry->overridden = false;
        }
    }

    if(event.type == SDL_MOUSEMOTION) {
        if(event.motion.state & SDL_BUTTON_LMASK) {
            for(IntChangeEntry *entry: left_mouse_vars_) {
                modify_int_entry(entry, -1, event.motion.xrel);
            }
        }
        if(event.motion.state & SDL_BUTTON_RMASK) {
            for(IntChangeEntry *entry: right_mouse_vars_) {
                modify_int_entry(entry, -1, event.motion.yrel);
            }
        }

    }
    else if(event.type == SDL_MOUSEBUTTONDOWN) {
        if(control && (event.button.button == SDL_BUTTON_LEFT ||
                       event.button.button == SDL_BUTTON_RIGHT )) {
            std::vector<IntChangeEntry*> *active;
            if(event.button.button == SDL_BUTTON_LEFT) {
                active = &left_mouse_vars_;
            }
            else {
                active = &right_mouse_vars_;
            }
            active->clear();
            for(std::list<IntChangeEntry>::iterator entry = int_changes_.begin();
                    entry != int_changes_.end(); ++entry) {
                if(entry->key_pressed) {
                    active->push_back(&*entry);
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
        for(std::list<IntChangeEntry>::iterator entry = int_changes_.begin();
                entry != int_changes_.end(); ++entry) {
            if(entry->key_pressed) {
                modify_int_entry(&*entry, -1, modify_value);
            }
        }
    }
    if(override_value != -1) {
        for(std::list<IntChangeEntry>::iterator entry = int_changes_.begin();
                entry != int_changes_.end(); ++entry) {
            if(entry->key_pressed) {
                entry->override_value *= 10;
                entry->override_value += override_value;
                entry->overridden = true;
            }
        }
    }
}

void InputManager::modify_int_entry(IntChangeEntry *entry, int override_value,
                                    int modify_value) {
    std::cout << "asdfasdf" << std::endl;
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

void InputManager::print_controls() {
    std::cout << std::endl << "Toggle Keys:" << std::endl;
    for(BoolTogglerEntry entry: bool_toggles_) {
        std::cout << "  "
                  << SDL_GetKeyName(entry.key)
                  << ": "
                  << entry.name
                  << std::endl;
    }
    std::cout << std::endl << "Function Keys:" << std::endl;
    for(FunctionCallerEntry entry: function_callers_) {
        std::cout << "  "
                  << SDL_GetKeyName(entry.key)
                  << ": "
                  << entry.name
                  << std::endl;
    }
    std::cout << std::endl << "Integer Keys:" << std::endl;
    for(IntChangeEntry entry: int_changes_) {
        std::cout << "  "
                  << SDL_GetKeyName(entry.key)
                  << ": "
                  << entry.name
                  << std::endl;
    }
}

void InputManager::remove_var_changer(SDL_Keycode key) {
    if(used_keys_.erase(key) == 0) {
        std::cout << "Attempted to remove key "
                  << SDL_GetKeyName(key) 
                  << " which wasn't added." << std::endl;
        return;
    }

    bool_toggles_.remove_if([key](BoolTogglerEntry b){ return b.key == key; });
    function_callers_.remove_if([key](FunctionCallerEntry f){ return f.key == key; });
    int_changes_.remove_if([key](IntChangeEntry i){ return i.key == key; });
}

void InputManager::reset() {
    left_mouse_vars_.clear();
    right_mouse_vars_.clear();
}
