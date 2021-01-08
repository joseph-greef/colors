
#include <climits>
#include <iostream>
#include <stdlib.h>

#include "ants.h"
#include "input_manager.h"

Ants::Ants(int width, int height)
    : Ruleset(width, height)
    , colony_pheromone_display_(0)
    , color_speed_(2)
    , current_tick_(0)
    , food_probability_(10)
    , num_colonies_(8)
    , num_food_for_child_(50)
    , rainbows_(width, height, 0)
    , rainbow_train_len_(256)
    , rainbow_view_(false)
    , starting_food_density_(1500)
    , e2_(rd_())
    , dist_(0, 1)
{
    rainbow_board_ = new int[width * height];
    world_ = new WorldEntry[width_ * height_];
    reset();
}

Ants::~Ants() {
    for(Colony *colony: colonies_) {
        delete colony;
    }
    for(Food *food: foods_) {
        delete food;
    }
    delete [] rainbow_board_;
    delete [] world_;
}

void Ants::add_colony(int num_ants) {
    colonies_.push_back(new Colony(width_, height_,
                                   rand() % width_, rand() % height_,
                                   generate_color()));
    colonies_.back()->add_ants(&ants_, num_ants);
}

uint32_t Ants::generate_color() {
    uint8_t r = 0;
    uint8_t b = 0;
    uint8_t g = 0;
    do {
        r = rand() % 245 + 5;
        g = rand() % 245 + 5;
        b = rand() % 245 + 5;
    } while(g > r + b || r + g + b < 200);

    return (r << 0) |
           (g << 8) |
           (b << 16);
}

void Ants::get_pixels(uint32_t *pixels) {
    if(rainbow_view_) {
        rainbows_.age_to_pixels(rainbow_board_, pixels);
    }
    else {
        if(colony_pheromone_display_ > 0 &&
           colony_pheromone_display_ <= static_cast<int>(colonies_.size()))
        {
            auto colonies_it = colonies_.begin();
            std::advance(colonies_it, colony_pheromone_display_ - 1);
            (*colonies_it)->draw_pheromones(pixels);
        }
        else {
            for(int j = 0; j < height_; j++) {
                for(int i = 0; i < width_; i++) {
                    pixels[j * width_ + i] = 0;
                }
            }
        }
        for(Colony *colony: colonies_) {
            colony->draw_self(pixels);
        }
        for(Ant *ant: ants_) {
            int offset = ant->y * width_ + ant->x;
            pixels[offset] = ant->colony->get_color();
        }
        for(Food *food: foods_) {
            int offset = food->y * width_ + food->x;
            pixels[offset] = 0x00FF00; //Green food
        }
    }
}

void Ants::print_rules() {
}

void Ants::reset() {
    for(Colony *colony: colonies_) {
        delete colony;
    }
    colonies_.clear();
    ants_.clear();
    for(Food *food: foods_) {
        delete food;
    }
    foods_.clear();

    restock_colonies(25);
    for(int i = 0; i < width_ * height_ / starting_food_density_; i++) {
        foods_.push_back(new Food(rand() % (width_ - 2) + 1, 
                                  rand() % (height_ - 2) + 1, 
                                  rand() % 50));
    }

    current_tick_ = 0;
    memset(rainbow_board_, 0, width_ * height_ * sizeof(int));
}

void Ants::restock_colonies(int num_ants) {
    while(static_cast<int>(colonies_.size()) < num_colonies_) {
        std::cout << "restocking" << std::endl;
        add_colony(num_ants);
    }
}

#ifdef USE_GPU
void Ants::start_cuda() {
}

void Ants::stop_cuda() {
}

#endif

void Ants::start() {
    std::cout << "Starting Ants" << std::endl;
    Ruleset::start();

    InputManager::add_bool_toggler(&rainbow_view_, SDL_SCANCODE_T, false, false,
                                   "(Ants) Toggle rainbow view");

    ADD_FUNCTION_CALLER(&Ants::reset, SDL_SCANCODE_E, false, false,
                        "(Ants) Reset simulation");
    ADD_FUNCTION_CALLER_W_ARGS(&Ants::add_colony, SDL_SCANCODE_R, false, false,
                               "(Ants) Add ant colony with random DNA", 5);

    //InputManager::add_int_changer(&colony_pheromone_display_, SDLK_a, 0, INT_MAX, "(Ants) Pheromone Display");
    //InputManager::add_int_changer(&num_colonies_, SDLK_s, 0, INT_MAX, "(Ants) Minimum Colonies");
    //InputManager::add_int_changer(&color_speed_, SDLK_d, 0, INT_MAX, "(Ants) Color Speed");
    //InputManager::add_int_changer(&rainbow_train_len_, SDLK_q, 0, INT_MAX, "(Ants) Trail Length");
    //InputManager::add_int_changer(&num_food_for_child_, SDLK_w, 0, INT_MAX, "(Ants) Num Food to Spawn Child");
    rainbows_.start();
}

void Ants::stop() {
    Ruleset::stop();

    InputManager::remove_var_changer(SDL_SCANCODE_T, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_E, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_R, false, false);

    //InputManager::remove_var_changer(SDLK_a);
    //InputManager::remove_var_changer(SDLK_s);
    //InputManager::remove_var_changer(SDLK_d);
    //InputManager::remove_var_changer(SDLK_q);
    rainbows_.stop();
}

void Ants::tick() {
    std::vector<Ant*> ant_to_remove;
    std::vector<Colony*> colonies_to_add;
    std::vector<Colony*> colonies_to_remove;
#ifdef USE_GPU
    cv::cuda::Stream cuda_stream;
#endif //USE_GPU

    //Move every ant
    for(Ant *ant: ants_) {
        if(!ant->colony->move_ant(ant)) {
            ant_to_remove.push_back(ant);
        }
        else {
            rainbow_board_[ant->y * width_ + ant->x] = rainbow_train_len_;
        }
    }
    //Delete the ones that died
    for(Ant *ant: ant_to_remove) {
        ants_.remove(ant);
    }

    if(current_tick_ % color_speed_ == 0) {
        for(int j = 0; j < height_; j++) {
            for(int i = 0; i < width_; i++) {
                if(rainbow_board_[j * width_ + i] > 0) {
                    rainbow_board_[j * width_ + i]--;
                }
            }
        }
    }

    //Update colony state
    for(Colony *colony: colonies_) {
        if(colony->get_num_ants() < 5) {
            colonies_to_remove.push_back(colony);
        }
        else {
            if(colony->get_num_food_collected() > num_food_for_child_) {
                colonies_to_add.push_back(colony->make_child());
            }

            for(Food *food: foods_) {
                colony->add_food_smell(food->x, food->y, 10);
            }
            for(Ant *ant: ants_) {
                if(!colony->owns_ant(ant)) {
                    colony->add_enemy_smell(ant->x, ant->y, 10);
                }
            }
#ifdef USE_GPU
            colony->queue_cuda_ops(cuda_stream);
#else
            colony->update_pheromones();
#endif //USE_GPU
        }
    }
#ifdef USE_GPU
    cuda_stream.waitForCompletion();
#endif //USE_GPU

    for(Colony *colony: colonies_to_add) {
        colonies_.push_back(colony);
        colonies_.back()->add_ants(&ants_, 5);
    }

    for(Colony *colony: colonies_to_remove) {
        for(Ant *ant: *colony->get_ants()) {
            ants_.remove(ant);
        }
        colonies_.remove(colony);
        delete colony;
    }

    //Check if we need to add more colonies, and init them with 5 ants
    restock_colonies(5);

    //Detect and handle events
    memset(world_, 0, width_*height_*sizeof(world_[0]));

    for(Food *food: foods_) {
        int offset = food->y * width_ + food->x;                            
        world_[offset].type = FoodType;
        world_[offset].ptr = food;
    }

    for(Colony *colony: colonies_) {
        int offset = colony->get_offset();                            
        world_[offset].type = ColonyType;
        world_[offset].ptr = colony;
    }

    ant_to_remove.clear();
    for(Ant *ant: ants_) {
        int offset = ant->y * width_ + ant->x;                            
        bool overwrite_world = true;
        if(world_[offset].type == FoodType) {
            Food *f = static_cast<Food*>(world_[offset].ptr);
            if(!ant->has_food && f->bites_left > 0) {
                f->bites_left -= 1;
                if(f->bites_left == 0) {
                    delete f;
                    foods_.remove(f);
                    foods_.push_back(new Food(rand() % (width_ - 2) + 1, 
                                              rand() % (height_ - 2) + 1, 
                                              rand() % 50));
                }
                ant->has_food = true;
                ant->steps_since_event = 0;
                ant->colony->add_food_smell(ant->x, ant->y, 10);
            }
        }
        else if(world_[offset].type == ColonyType) {
            Colony *c = static_cast<Colony*>(world_[offset].ptr);
            if(ant->has_food && c->owns_ant(ant)) {
                ant->has_food = false;
                ant->enemy_seen = false;
                ant->colony->add_ants(&ants_, 1);
                ant->colony->food_collected();
                ant->steps_since_event = 0;
            }
        }
        else if(world_[offset].type == AntType) {
            Ant *a = static_cast<Ant*>(world_[offset].ptr);
            if(a->colony != ant->colony) {
                float roll1 = dist_(e2_);
                float roll2 = dist_(e2_);
                if(a->colony->enemy_encountered(a, ant, roll1, roll2)) {
                    ant_to_remove.push_back(ant);
                    a->has_food = true;
                    a->enemy_seen = true;
                    a->steps_since_event = 0;
                    overwrite_world = false;
                }
                if(ant->colony->enemy_encountered(ant, a, roll2, roll1)) {
                    ant_to_remove.push_back(a);
                    ant->has_food = true;
                    ant->enemy_seen = true;
                    ant->steps_since_event = 0;
                }
            }
        }
        if(overwrite_world) {
            world_[offset].type = AntType;
            world_[offset].ptr = ant;
        }
    }
    for(Ant *ant: ant_to_remove) {
        ants_.remove(ant);
        delete ant;
    }

    current_tick_++;
} //tick()
