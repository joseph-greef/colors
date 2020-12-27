
#ifndef _INPUT_MANAGER_H
#define _INPUT_MANAGER_H

#include <SDL.h>

#include <functional>
#include <list>
#include <set>
#include <string>
#include <vector>

using std::placeholders::_1;
using std::placeholders::_2;
#define ADD_FUNCTION_CALLER_W_ARGS(func, key, name, ...) \
    InputManager::add_function_caller( \
        std::bind((func), this, _1, _2, __VA_ARGS__), \
        key, name)

#define ADD_FUNCTION_CALLER(func, key, name) \
    InputManager::add_function_caller( \
        std::bind((func), this, _1, _2), \
        key, name)

typedef void (*ManagerFunc)(bool control, bool shift);

struct VarChangeEntry {
    VarChangeEntry(SDL_Keycode key_, std::string name_)
        : key(key_)
        , name(name_)
    {}

    SDL_Keycode key;
    std::string name;

    bool operator <(const VarChangeEntry & otherVar) const {
        return key < otherVar.key;
    }
};

struct BoolTogglerEntry: public VarChangeEntry {
    BoolTogglerEntry(SDL_Keycode key_, std::string name_, bool *variable_)
        : VarChangeEntry(key_, name_)
        , variable(variable_)
    {}

    bool *variable;
};

struct FunctionCallerEntry: public VarChangeEntry {
    FunctionCallerEntry(SDL_Keycode key_, std::string name_, 
                        std::function<void(bool, bool)> function_)
        : VarChangeEntry(key_, name_)
        , function(function_)
    {}

    std::function<void(bool, bool)> function;
};

struct IntChangeEntry: public VarChangeEntry {
    IntChangeEntry(SDL_Keycode key_, std::string name_, int max_value,
                     int min_value, int *variable)
        : VarChangeEntry(key_, name_)
        , key_pressed(0)
        , max_value(max_value)
        , min_value(min_value)
        , override_value(0)
        , overridden(false)
        , variable(variable)
    {}

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

        static std::list<BoolTogglerEntry*> bool_toggles_;
        static std::list<FunctionCallerEntry*> function_callers_;
        static std::list<IntChangeEntry*> int_changes_;

        static std::vector<IntChangeEntry*> left_mouse_vars_;
        static std::vector<IntChangeEntry*> right_mouse_vars_;

        static bool check_and_insert_key(SDL_Keycode key, std::string name);
        static void handle_bool_events(SDL_Event event, bool control, bool shift);
        static void handle_function_events(SDL_Event event, bool control, bool shift);
        static void handle_int_events(SDL_Event event, bool control, bool shift);
        static void modify_int_entry(IntChangeEntry *entry, int override_value,
                                     int modify_entry);
    public:
        static void add_bool_toggler(bool *variable, SDL_Keycode key, std::string name);

        static void add_function_caller(std::function<void(bool, bool)> function,
                                        SDL_Keycode key, std::string name);
        static void add_int_changer(int *variable, SDL_Keycode key,
                                    int min_value, int max_value, std::string name);
        static void handle_input(SDL_Event event, bool control, bool shift);
        static void print_controls();
        static void remove_var_changer(SDL_Keycode key);
        static void reset();
};

#endif //_INPUT_MANAGER_H

