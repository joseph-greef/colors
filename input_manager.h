
#ifndef _INPUT_MANAGER_H
#define _INPUT_MANAGER_H

#include <SDL.h>

#include <set>
#include <string>
#include <vector>


struct VarChangeEntry{
    SDL_Keycode key;
    std::string name;

    bool operator <(const VarChangeEntry & otherVar) const {
        return key < otherVar.key;
    }
};

struct IntChangeEntry: public VarChangeEntry{
    bool key_pressed;
    int max_value;
    int min_value;
    int override_value;
    bool overridden;
    int *variable;
};

class InputManager {
    private:
        static std::set<SDL_Keycode> used_keys_;

        static std::set<IntChangeEntry*> int_changes_;
        static std::vector<IntChangeEntry*> left_mouse_vars_;
        static std::vector<IntChangeEntry*> right_mouse_vars_;

        static void handle_int_events(SDL_Event event, bool control, bool shift);
        static void modify_int_entry(IntChangeEntry *entry, int override_value,
                                     int modify_entry);
    public:
        static void add_int_changer(int *variable, SDL_Keycode key,
                                    int min_value, int max_value, std::string name);
        static void handle_input(SDL_Event event, bool control, bool shift);
        static void print_controls();
        static void remove_var_changer(SDL_Keycode key);
        static void reset();
};

#endif //_INPUT_MANAGER_H

