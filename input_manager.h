
#ifndef _INPUT_MANAGER_H
#define _INPUT_MANAGER_H

#include <SDL.h>

#include <set>
#include <string>


typedef struct var_change_entry {
    SDL_Keycode key;
    bool key_pressed;
    int max_value;
    int min_value;
    int multiplier;
    std::string name;
    int *variable;
} VarChangeEntry;

class InputManager {
    private:
        static bool compare_entries(VarChangeEntry *a, VarChangeEntry *b);

        static std::set<SDL_Keycode> used_keys_;
        static std::set<VarChangeEntry*, 
                        decltype(InputManager::compare_entries)*> int_changes_;
    public:
        static void add_var_changer(int *variable, SDL_Keycode key, int multiplier,
                                    int min_value, int max_value, std::string name);
        static void handle_input(SDL_Event event, bool control, bool shift);
        static void remove_var_changer(SDL_Keycode key);
};

#endif //_INPUT_MANAGER_H

