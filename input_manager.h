
#ifndef _INPUT_MANAGER_H
#define _INPUT_MANAGER_H

#include <functional>
#include <list>
#include <set>
#include <string>
#include <vector>

#include "SDL2/SDL.h"


using std::placeholders::_1;
using std::placeholders::_2;
#define ADD_FUNCTION_CALLER_W_ARGS(func, func_type, scancode, control, shift, owner_name, description, ...) \
    InputManager::add_input(static_cast<func_type>(std::bind((func), this, __VA_ARGS__)), \
                            scancode, control, shift, owner_name, description)

#define ADD_FUNCTION_CALLER(func, scancode, control, shift, owner_name, description) \
    InputManager::add_input(std::bind((func), this), \
                            scancode, control, shift, owner_name, description)


namespace FunctionType {
enum FunctionType {
    None = 0,
    Void,
    Int,
    String,
};
}


namespace ManagerMode {
enum ManagerMode {
    Normal,
    IntAccumulator,
    StringAccumulator,
};
}


typedef std::function<void(void)> VoidFunc;
typedef std::function<int(int, int)> IntFunc;
typedef std::function<std::string(std::string)> StringFunc;


struct ComboFunction {
    FunctionType::FunctionType func_type;
    VoidFunc void_func;
    IntFunc int_func;
    StringFunc string_func;
    std::string description;
    std::string owner_name;
};


struct KeyFunction {
    ComboFunction no_mod;
    ComboFunction control;
    ComboFunction shift;
    ComboFunction control_shift;
};


struct IntEntry{
    IntEntry(int max_value, int min_value, int *variable, SDL_Scancode scancode,
             bool control, bool shift)
        : max_value(max_value)
        , min_value(min_value)
        , variable(variable)
        , scancode(scancode)
        , control(control)
        , shift(shift)
    {}

    int max_value;
    int min_value;
    int *variable;

    SDL_Scancode scancode;
    bool control;
    bool shift;

    bool operator ==(const IntEntry & other_entry) const {
        return scancode == other_entry.scancode &&
               control == other_entry.control &&
               shift == other_entry.shift;
    }
};


class InputManager {
    private:
        static std::set<ComboFunction*> active_int_combos_;
        static ComboFunction* active_string_combo_;
        static int int_accumulator_;
        static std::list<IntEntry> int_entries_;
        static KeyFunction key_functions_[SDL_NUM_SCANCODES];
        static ManagerMode::ManagerMode mode_;
        static bool reset_pending_;

        static std::list<ComboFunction*> mouse_left_combos_;
        static FunctionType::FunctionType mouse_left_mode_;

        static std::list<ComboFunction*> mouse_right_combos_;
        static FunctionType::FunctionType mouse_right_mode_;

        static std::string string_accumulator_;


        static ComboFunction* get_combo_func(SDL_Scancode scancode, bool control,
                                             bool shift);

        static int modify_int(IntEntry *entry, int override_value, int modify_entry);
        static void reset();
        static void toggle_bool(bool *var);

    public:
        static void add_input(VoidFunc func,
                              SDL_Scancode scancode, bool control, bool shift,
                              std::string owner_name, std::string description);
        static void add_input(IntFunc func,
                              SDL_Scancode scancode, bool control, bool shift,
                              std::string owner_name, std::string description);
        static void add_input(StringFunc func,
                              SDL_Scancode scancode, bool control, bool shift,
                              std::string owner_name, std::string description);

        static void add_int_changer(int *variable, SDL_Scancode,
                                    bool control, bool shift,
                                    int min_value, int max_value,
                                    std::string owner_name, std::string description);
        static void add_bool_toggler(bool *variable, SDL_Scancode scancode,
                                     bool control, bool shift,
                                     std::string owner_name, std::string description);

        static void handle_input(SDL_Event event);
        static void print_controls();
        static void remove_var_changer(SDL_Scancode scancode, bool control, bool shift);
        static void trigger_reset();
};

#endif //_INPUT_MANAGER_H

