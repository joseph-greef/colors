#ifndef _ANTS_COLONY_H
#define _ANTS_COLONY_H

#include <list>
#include <random>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"

class Colony;

#include "ant.h"

struct ColonyDNA {
    float aggression_;
    float enemy_encounter_amount_;
    float enemy_signal_strength_;
    float enemy_smooth_amount_;
    float food_signal_strength_;
    float food_smooth_amount_;
    float home_signal_strength_;
    float home_smell_amount_;
    float home_smooth_amount_;
    float pheromone_decay_rate_;
    float randomness_;
    int enemy_blur_size_;
    int food_blur_size_;
    int home_blur_size_;
    int max_signal_steps_;
    int max_total_steps_;
};


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
        uint32_t num_food_;

        ColonyDNA DNA_;
        std::random_device rd_;
        std::mt19937 e2_;
        std::uniform_real_distribution<> dist_full_;
        std::uniform_real_distribution<> dist_positive_;

#ifdef USE_GPU
        cv::cuda::GpuMat enemy_mat_;
        cv::cuda::GpuMat enemy_mat_buffer_;
        cv::Ptr<cv::cuda::Filter> enemy_gauss_;

        cv::cuda::GpuMat food_mat_;
        cv::cuda::GpuMat food_mat_buffer_;
        cv::Ptr<cv::cuda::Filter> food_gauss_;

        cv::cuda::GpuMat home_mat_;
        cv::cuda::GpuMat home_mat_buffer_;
        cv::Ptr<cv::cuda::Filter> home_gauss_;
#endif //USE_GPU

        float *enemy_pheromones_;
        float *enemy_pheromones_buffer_;

        float *food_pheromones_;
        float *food_pheromones_buffer_;

        float *home_pheromones_;
        float *home_pheromones_buffer_;

    public:
        Colony(int width, int height, int x, int y, uint32_t color);
        Colony(int width, int height, uint32_t color, ColonyDNA *DNA,
               Ant *starting_ant);
        Colony(int width, int height, int x, int y, uint32_t color, std::string dna_string);
        ~Colony();
        void add_ants(std::list<Ant*> *ants, int number);
        void add_enemy_smell(int x, int y, float amount);
        void add_food_smell(int x, int y, float amount);
        void draw_pheromones(uint32_t *pixels);
        void draw_self(uint32_t *pixels);
        bool enemy_encountered(Ant *ant, Ant *enemy_ant,
                               float roll, float enemy_roll);
        void food_collected();
        std::list<Ant*> *get_ants();
        float get_aggression();
        uint32_t get_color();
        std::string get_dna_string();
        int get_offset();
        uint32_t get_num_ants();
        int get_num_food_collected();
        int get_x();
        int get_y();
        bool move_ant(Ant *ant);
        Colony* make_child();
        bool owns_ant(Ant *ant);
        void update_pheromones();

#ifdef USE_GPU
        void queue_cuda_ops(cv::cuda::Stream stream);
#endif //USE_GPU

};

#endif //_ANTS_COLONY_H

