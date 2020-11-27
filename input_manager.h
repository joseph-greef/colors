
#ifndef _INPUT_MANAGER_H
#define _INPUT_MANAGER_H

#include <SDL.h>

#include <set>
#include <string>
#include <vector>


typedef struct var_change_entry {
    SDL_Keycode key;
    bool key_pressed;
    int max_value;
    int min_value;
    int override_value;
    bool overridden;
    std::string name;
    int *variable;
} VarChangeEntry;

class InputManager {
    private:
        static bool compare_entries(VarChangeEntry *a, VarChangeEntry *b);

        static std::set<SDL_Keycode> used_keys_;
        static std::set<VarChangeEntry*, 
                        decltype(InputManager::compare_entries)*> int_changes_;
        static std::vector<VarChangeEntry*> left_mouse_vars_;
        static std::vector<VarChangeEntry*> right_mouse_vars_;

        static void modify_entry(VarChangeEntry *entry, int override_value, int modify_entry);
    public:
        static void add_var_changer(int *variable, SDL_Keycode key,
                                    int min_value, int max_value, std::string name);
        static void handle_input(SDL_Event event, bool control, bool shift);
        static void remove_var_changer(SDL_Keycode key);
        static void reset();
};

#endif //_INPUT_MANAGER_H

