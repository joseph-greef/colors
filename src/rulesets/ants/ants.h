#ifndef _ANTS_ANTS_H
#define _ANTS_ANTS_H

#include <list>
#include <random>

#include "ruleset.h"
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
        std::list<Colony*> colonies_;
        std::list<Ant*> ants_;
        std::list<Food*> foods_;
        int *rainbow_board_;
        int colony_pheromone_display_;
        int color_speed_;
        int current_tick_;
        int food_probability_;
        int num_colonies_;
        int num_food_for_child_;
        Rainbows rainbows_;
        int rainbow_train_len_;
        bool rainbow_view_;
        int starting_food_density_;
        std::random_device rd_;
        std::mt19937 e2_;
        std::uniform_real_distribution<> dist_;

        WorldEntry *world_;

        void add_colony(int num_ants);
        uint32_t generate_color();
        void reset();
        void restock_colonies(int num_ants);

        void start_cuda();
        void stop_cuda();

    public:
        Ants(int width, int height);
        ~Ants();

        BoardType::BoardType board_get_type();
        BoardType::BoardType board_set_type();
        void* get_board();
        std::string get_name();
        void get_pixels(uint32_t *pixels);
        std::string get_rule_string();
        void load_rule_string(std::string rules);
        void print_human_readable_rules();
        void set_board(void *new_board);
        void start();
        void stop();
        void tick();
};

#endif //_ANTS_ANTS_H
