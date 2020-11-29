#ifndef _ANTS_ANTS_H
#define _ANTS_ANTS_H

#include <list>
#include <random>
#include <vector>

#include "../ruleset.h"
#include "colony.h"
#include "food.h"

enum WorldEntryType {
    NoneType,
    FoodType,
    AntType,
    ColonyType,
};

struct WorldEntry {
    WorldEntryType type;
    //TODO: Real polymorphism
    void *ptr;
    int index;
};
    

class Ants : public Ruleset {
    private:
        std::vector<Colony*> colonies_;
        std::list<Ant*> ants_;
        std::list<Food*> foods_;
        int colony_pheromone_display_;
        int food_probability_;
        int num_colonies_;
        int starting_food_density_;
        std::random_device rd_;
        std::mt19937 e2_;
        std::uniform_real_distribution<> dist_;

        WorldEntry *world_;

        void reset();
#ifdef USE_GPU
        void start_cuda();
        void stop_cuda();
#endif

        static uint32_t colony_colors[];
        static int max_colonies;

    public:
        Ants(int width, int height);
        ~Ants();
        void get_pixels(uint32_t *pixels);
        void handle_input(SDL_Event event, bool control, bool shift);
        void print_controls();
        void print_rules();
        void start();
        void stop();
        void tick();
};

#endif //_ANTS_ANTS_H
