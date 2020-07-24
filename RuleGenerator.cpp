#include "RuleGenerator.h"



RuleGenerator::RuleGenerator(unsigned int new_max_seeds)
{
    max_seeds = new_max_seeds;
}

RuleGenerator::~RuleGenerator()
{
    //dtor
}

float* RuleGenerator::generate_one_mean() {
    float probabilities[18];
    float *to_return = (float*) malloc(sizeof(float) * 18);
    for(int i = 0; i < 18; i++) {
        probabilities[i] = 0;
    }

    for(std::deque<float*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
        for(int k = 0; k < 18; k++) {
            probabilities[k] += (*it)[k];
        }
    }

    for(int i = 0; i < 18; i++) {
        probabilities[i] = probabilities[i] / seeds.size();
    }

    for(int i = 0; i < 18; i++) {
        to_return[i] = probabilities[i] >= (random_float() - 0.1) ? 1 : 0;
    }

    return to_return;
}

float* RuleGenerator::generate_one_mean_float() {
    float probabilities[18];
    float *to_return = (float*) malloc(sizeof(float) * 18);
    for(int i = 0; i < 18; i++) {
        probabilities[i] = 0;
    }

    for(std::deque<float*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
        for(int k = 0; k < 18; k++) {
            probabilities[k] += (*it)[k];
        }
    }

    for(int i = 0; i < 18; i++) {
        probabilities[i] = probabilities[i] / seeds.size();
    }

    int i;
    for(i = 0; i < 9; i++) {
        to_return[i] = probabilities[i] + ((random_float() < 0.5) ? -0.1 : 0.1);
    }

    for(; i < 18; i++) {
        to_return[i] = probabilities[i] >= (random_float() - 0.1) ? 1 : 0;
    }

    return to_return;
}

void RuleGenerator::add_seed(float* seed) {
    seeds.push_back(seed);
    if(seeds.size() > max_seeds) {
        free(seeds.front());
        seeds.pop_front();
    }
}

void RuleGenerator::print_seeds() {
    for(std::deque<float*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
        print_array(*it);
    }
}

void RuleGenerator::print_array(float *to_print) {
    std::cout << "[ ";
    for(int i = 0; i < 18; i++) {
        std::cout << to_print[i] << " ";
    }
    std::cout << "]" << std::endl;
    return;
}

float RuleGenerator::random_float() {
    float f = (float)rand() / RAND_MAX;
    return f;
}
