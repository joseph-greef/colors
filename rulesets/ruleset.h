#ifndef _RULESET_H
#define _RULESET_H

#include <SDL.h>
#include <stdint.h>
#include <vector>

enum NeighborhoodType {
    VonNeuman,
    Moore,
};

typedef struct var_change_entry {
    int *variable;
    SDL_Keycode key;
    int multiplier;
    bool key_pressed;
} VarChangeEntry;

class Ruleset {
    private:
        std::vector<VarChangeEntry*> var_changes_;
    protected:
        int height_;
        int width_;

    public:
        Ruleset(int width, int height);
        ~Ruleset();

        virtual void get_pixels(uint32_t *pixels) = 0;
        virtual void handle_input(SDL_Event event, bool control, bool shift) = 0;
        virtual void print_rules() = 0;
        virtual void tick() = 0;

        void add_var_changer(int *variable, SDL_Keycode key, int multiplier);
        int get_num_alive_neighbors(int *board, int x, int y, int radius,
                                    NeighborhoodType type);
        void handle_var_changers(SDL_Event event, bool control, bool shift);
};

#endif //_RULE_SET_H
