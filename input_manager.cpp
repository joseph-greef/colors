
#include <climits>
#include <iomanip>
#include <iostream>

#include "input_manager.h"

/*
Fully static object, array of KeyBehaviors or something that have control,
shift and control+shift behaviors. These can call functions, or turn the whole
machine to accumulator mode where it will record strings or numbers.

Need to figure out incrementing.

Mouse control? Register mouse callbacks?
*/

std::set<ComboFunction*> InputManager::active_int_combos_ =
        std::set<ComboFunction*>();
ComboFunction* InputManager::active_string_combo_ = NULL;
int InputManager::int_accumulator_ = INT_MIN;
std::list<IntEntry> InputManager::int_entries_ = std::list<IntEntry>();
KeyFunction InputManager::key_functions_[SDL_NUM_SCANCODES] = { FunctionType::None };
ManagerMode::ManagerMode InputManager::mode_ = ManagerMode::Normal;

std::list<ComboFunction*> InputManager::mouse_left_combos_ = 
        std::list<ComboFunction*>();
FunctionType::FunctionType InputManager::mouse_left_mode_ = FunctionType::None;

std::list<ComboFunction*> InputManager::mouse_right_combos_ = 
        std::list<ComboFunction*>();
FunctionType::FunctionType InputManager::mouse_right_mode_ = FunctionType::None;

std::string InputManager::string_accumulator_ = std::string();

void InputManager::add_bool_toggler(bool *variable, SDL_Scancode scancode,
                                    bool control, bool shift,
                                    std::string owner_name, std::string description) {
    add_input(std::bind(InputManager::toggle_bool, variable), scancode,
              control, shift, owner_name, description);
}

void InputManager::add_input(VoidFunc func,
                             SDL_Scancode scancode, bool control, bool shift,
                             std::string owner_name, std::string description) {
    ComboFunction *combo = get_combo_func(scancode, control, shift);
    if(combo->func_type == FunctionType::None) {
        combo->func_type = FunctionType::Void;
        combo->void_func = func;
        combo->description = description;
        combo->owner_name = owner_name;
    }
    else {
        const char *key_name = SDL_GetKeyName(
                SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(scancode)));
        std::cout << key_name << " " << control << " " << shift <<
                     " is already registered. Not double registering." << std::endl;
        std::cout << "The current registration is " << combo->description << ". "
                     "The attempted registration is  " << description << std::endl;
    }
}

void InputManager::add_input(IntFunc func,
                             SDL_Scancode scancode, bool control, bool shift,
                             std::string owner_name, std::string description) {
    ComboFunction *combo = get_combo_func(scancode, control, shift);
    if(combo->func_type == FunctionType::None) {
        combo->func_type = FunctionType::Int;
        combo->int_func = func;
        combo->description = description;
        combo->owner_name = owner_name;
    }
    else {
        const char *key_name = SDL_GetKeyName(
                SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(scancode)));
        std::cout << key_name << " " << control << " " << shift <<
                     " is already registered. Not double registering." << std::endl;
        std::cout << "The current registration is " << combo->description << ". "
                     "The attempted registration is  " << description << std::endl;
    }
}

void InputManager::add_input(StringFunc func,
                             SDL_Scancode scancode, bool control, bool shift,
                             std::string owner_name, std::string description) {

    ComboFunction *combo = get_combo_func(scancode, control, shift);
    if(combo->func_type == FunctionType::None) {
        combo->func_type = FunctionType::String;
        combo->string_func = func;
        combo->description = description;
        combo->owner_name = owner_name;
    }
    else {
        const char *key_name = SDL_GetKeyName(
                SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(scancode)));
        std::cout << key_name << " " << control << " " << shift <<
                     " is already registered. Not double registering." << std::endl;
        std::cout << "The current registration is " << combo->description << ". "
                     "The attempted registration is " << description << std::endl;
    }
}

void InputManager::add_int_changer(int *variable, SDL_Scancode scancode,
                                   bool control, bool shift,
                                   int min_value, int max_value,
                                   std::string owner_name, std::string description) {
    IntEntry entry(max_value, min_value, variable, scancode, control, shift);
    int_entries_.push_back(entry);

    auto perm_entry = find(int_entries_.begin(), int_entries_.end(), entry);
    
    add_input(std::bind(InputManager::modify_int, &*perm_entry, _1, _2), scancode,
              control, shift, owner_name, description);
}

ComboFunction* InputManager::get_combo_func(SDL_Scancode scancode,
                                            bool control, bool shift) {
    ComboFunction *combo = &key_functions_[scancode].no_mod;
    if(control && shift) {
        combo = &key_functions_[scancode].control_shift;
    }
    else if(control) {
        combo = &key_functions_[scancode].control;
    }
    else if(shift) {
        combo = &key_functions_[scancode].shift;
    }

    return combo;
}

void InputManager::handle_input(SDL_Event event) {
    static bool control = false;
    static bool shift = false;
    int mul;

    if(event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        control = event.key.keysym.mod & KMOD_CTRL;
        shift = event.key.keysym.mod & KMOD_SHIFT;
    }

    mul = (control ? 2 : 1) *
          (shift ? 5 : 1);

    if(event.type == SDL_MOUSEMOTION) {
        if(event.motion.state & SDL_BUTTON_LMASK) {
            if(mouse_left_mode_ == FunctionType::Int) {
                for(ComboFunction *combo: mouse_left_combos_) {
                    combo->int_func(INT_MIN, event.motion.xrel);
                }
            }
        }
        if(event.motion.state & SDL_BUTTON_RMASK) {
            if(mouse_right_mode_ == FunctionType::Int) {
                for(ComboFunction *combo: mouse_right_combos_) {
                    //Positive y is down, negate yrel to the control make sense
                    combo->int_func(INT_MIN, -event.motion.yrel);
                }
            }
        }
    }
    else if(event.type == SDL_KEYDOWN) {
        if(event.key.keysym.scancode == SDL_SCANCODE_BACKSPACE) {
            std::cout << "Clearing all active controls" << std::endl;
            reset();
        }
    }
    if((mode_ == ManagerMode::Normal || mode_ == ManagerMode::IntAccumulator) &&
            event.type == SDL_KEYDOWN) {
        ComboFunction *combo = get_combo_func(event.key.keysym.scancode,
                                              control, shift); 
        switch(combo->func_type) {
            case FunctionType::Void:
                combo->void_func();
                break;
            case FunctionType::Int:
            {
                auto ret = active_int_combos_.insert(combo);
                if(!ret.second) {
                    active_int_combos_.erase(combo);
                    std::cout << "Deactivated " << combo->description << std::endl;
                }
                else {
                    std::cout << "Activated " << combo->description << std::endl;
                    mode_ = ManagerMode::IntAccumulator;
                }

                if(active_int_combos_.empty()) {
                    reset();
                }
                return;
            }
            case FunctionType::String:
                if(mode_ == ManagerMode::Normal) {
                    mode_ = ManagerMode::StringAccumulator;
                    active_string_combo_ = combo;
                    std::cout << "Activated " << combo->description << std::endl;
                }
                else {
                    std::cout << "Must be in mormal mode to activate " <<
                                 combo->description << std::endl;
                }
                return;
        }


    }

    if(mode_ == ManagerMode::IntAccumulator) {
        int modify_value = 0;
        if(event.type == SDL_KEYDOWN) {
            if(event.key.keysym.scancode == SDL_SCANCODE_RETURN ||
               event.key.keysym.scancode == SDL_SCANCODE_KP_ENTER) {
                for(ComboFunction *combo: active_int_combos_) {
                    combo->int_func(int_accumulator_, 0);
                }
                int_accumulator_ = INT_MIN;
                return;
            }
            else if(event.key.keysym.sym >= SDLK_KP_1 && 
                    event.key.keysym.sym <= SDLK_KP_0) {
                //order is KP_1, KP_2 ... KP_0
                int num_val = event.key.keysym.sym - SDLK_KP_1 + 1;
                num_val = (num_val % 10);
                if(int_accumulator_ == INT_MIN) {
                    int_accumulator_ = 0;
                }
                int_accumulator_ = int_accumulator_ * 10 + num_val;
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
        else if(event.type == SDL_MOUSEBUTTONDOWN) {
            if(control) {
                if(event.button.button == SDL_BUTTON_LEFT) {
                    mouse_left_combos_.clear();
                    mouse_left_mode_ = FunctionType::Int;
                    for(ComboFunction *combo: active_int_combos_) {
                        mouse_left_combos_.push_back(combo);
                    }
                }
                else if(event.button.button == SDL_BUTTON_RIGHT) {
                    mouse_right_combos_.clear();
                    mouse_right_mode_ = FunctionType::Int;
                    for(ComboFunction *combo: active_int_combos_) {
                        mouse_right_combos_.push_back(combo);
                    }
                }
            }
            else if(shift) {
                if(event.button.button == SDL_BUTTON_LEFT) {
                    mouse_left_combos_.clear();
                    mouse_left_mode_ = FunctionType::None;
                }
                else if(event.button.button == SDL_BUTTON_RIGHT) {
                    mouse_right_combos_.clear();
                    mouse_right_mode_ = FunctionType::None;
                }
            }
        }

        if(modify_value != 0) {
            for(ComboFunction *combo: active_int_combos_) {
                combo->int_func(INT_MIN, modify_value);
            }
        }
    }
    if(mode_ == ManagerMode::StringAccumulator) {
        if(event.type == SDL_KEYDOWN) {
            const char *key_name = SDL_GetKeyName(
                    SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(
                            event.key.keysym.scancode)));
            if(event.key.keysym.scancode == SDL_SCANCODE_RETURN ||
               event.key.keysym.scancode == SDL_SCANCODE_KP_ENTER) {
                active_string_combo_->string_func(string_accumulator_);
                reset();
            }
            else if(strlen(key_name) == 1) {
                string_accumulator_ += *key_name;
            }
        }
    }
}

int InputManager::modify_int(IntEntry *entry, int override_value,
                              int modify_value) {
    if(override_value != INT_MIN) {
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
    
    std::cout << get_combo_func(entry->scancode, entry->control, entry->shift)->description 
              << ": " 
              << *(entry->variable) 
              << std::endl;

    return *(entry->variable);
}

void InputManager::print_controls() {
    struct RuleEntry {
        std::string owner_name;
        std::string description;
        std::string key_name;
        bool control;
        bool shift;

        RuleEntry(std::string owner_name, std::string description,
                  const char *key_name, bool control, bool shift)
            : owner_name(owner_name)
            , description(description)
            , key_name(key_name)
            , control(control)
            , shift(shift)
        {}

        bool operator <(RuleEntry &other) {
            if(owner_name != other.owner_name) {
                return owner_name < other.owner_name;
            }
            else {
                return key_name < other.key_name;
            }
        }
    };
    std::vector<RuleEntry> to_print;

    for(int i = 0; i < SDL_NUM_SCANCODES; i++) {
        const char *key_name = SDL_GetKeyName(
                SDL_GetKeyFromScancode(static_cast<SDL_Scancode>(i)));
        if(key_functions_[i].no_mod.func_type != FunctionType::None) {
            to_print.push_back(RuleEntry(key_functions_[i].no_mod.owner_name,
                                         key_functions_[i].no_mod.description,
                                         key_name, false, false));
        }
        if(key_functions_[i].control.func_type != FunctionType::None) {
            to_print.push_back(RuleEntry(key_functions_[i].control.owner_name,
                                         key_functions_[i].control.description,
                                         key_name, true, false));
        }
        if(key_functions_[i].shift.func_type != FunctionType::None) {
            to_print.push_back(RuleEntry(key_functions_[i].shift.owner_name,
                                         key_functions_[i].shift.description,
                                         key_name, false, true));
        }
        if(key_functions_[i].control_shift.func_type != FunctionType::None) {
            to_print.push_back(RuleEntry(key_functions_[i].control_shift.owner_name,
                                         key_functions_[i].control_shift.description,
                                         key_name, true, true));
        }
    }

    std::sort(to_print.begin(), to_print.end());

    std::cout << "           |   | S |" << std::endl;
    std::cout << "           | C | H |" << std::endl;
    std::cout << "           | T | I |" << std::endl;
    std::cout << "           | R | F |" << std::endl;
    std::cout << "    Key    | L | T |   Owner   | Description" << std::endl;
    std::cout << "-----------+---+---+-----------+------------" << std::endl;

    for(RuleEntry entry: to_print) {
        std::cout << std::setw(10) << entry.key_name << " | " << 
                     (entry.control ? "*" : " ") << " | " <<
                     (entry.shift ? "*" : " ") << " | " <<
                     std::setw(9) << entry.owner_name << 
                     " | " << entry.description << std::endl;
    }

}

void InputManager::remove_var_changer(SDL_Scancode scancode, bool control, bool shift) {
    ComboFunction *combo = get_combo_func(scancode, control, shift);
    IntEntry tmp_entry(0, 0, NULL, scancode, control, shift);

    combo->func_type = FunctionType::None;
    int_entries_.remove(tmp_entry);
}

void InputManager::reset() {
    active_int_combos_.clear();
    int_accumulator_ = INT_MIN;

    active_string_combo_ = NULL;
    string_accumulator_ = "";

    mode_ = ManagerMode::Normal;
}

void InputManager::toggle_bool(bool *var) {
    *var = !*var;
}

