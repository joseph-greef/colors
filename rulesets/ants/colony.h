#ifndef _ANTS_COLONY_H
#define _ANTS_COLONY_H

#include <list>
#include <random>

class Colony;

#include "ant.h"

class Colony {
    private:
        //Info variables
        int height_;
        int width_;
        int x_;
        int y_;
        int colony_number_;
        uint32_t color_;
        std::list<Ant*> ants_;
        //DNA variables
        float aggression_;
        float enemy_encounter_amount_;
        float enemy_signal_strength_;
        float food_signal_strength_;
        float home_signal_strength_;
        float home_smell_amount_;
        float randomness_;
        int max_signal_steps_;
        int max_total_steps_;

        std::random_device rd_;
        std::mt19937 e2_;
        std::uniform_real_distribution<> dist_;

        float *food_pheromones_;
        float *food_pheromones_buffer_;

        float *home_pheromones_;
        float *home_pheromones_buffer_;

        float *enemy_pheromones_;
        float *enemy_pheromones_buffer_;
    public:
        Colony(int width, int height, int x, int y, uint32_t color);
        ~Colony();
        void add_ants(std::list<Ant*> *ants, int number);
        void add_enemy_smell(int x, int y, float amount);
        void add_food_smell(int x, int y, float amount);
        void draw_pheromones(uint32_t *pixels);
        void draw_self(uint32_t *pixels);
        bool enemy_encountered(Ant *ant, Ant *enemy_ant,
                               float roll, float enemy_roll);
        std::list<Ant*> *get_ants();
        float get_aggression();
        uint32_t get_color();
        int get_offset();
        int get_x();
        int get_y();
        bool move_ant(Ant *ant);
        bool owns_ant(Ant *ant);
        void update_pheromones();

};

#endif //_ANTS_COLONY_H

