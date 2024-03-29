
#include <climits>
#include <iostream>
#include <stdlib.h>

#include "ants.h"
#include "input_manager.h"

Ants::Ants(int width, int height)
    : Ruleset()
    , colony_pheromone_display_(0)
    , color_speed_(2)
    , current_tick_(0)
    , food_probability_(10)
    , num_colonies_(8)
    , num_food_for_child_(50)
    , rainbows_(0)
    , rainbow_train_len_(256)
    , rainbow_view_(false)
    , starting_food_density_(1500)
    , e2_(rd_())
    , dist_(0, 1)
    , w_(width)
    , h_(height)
{
    rainbow_board_ = new int[width * height];
    world_ = new WorldEntry[w_ * h_];
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

/*
 * Cuda Functions:
 */
void Ants::start_cuda() {
}

void Ants::stop_cuda() {
}

/*
 * Buffer Copy Functions:
 */
std::set<std::size_t> Ants::buffer_types_provided() {
    std::set<std::size_t> buffers = { INT_BUFFER };
    return buffers;
}

std::size_t Ants::select_buffer_type(std::set<std::size_t> types) {
    return NOT_COMPATIBLE;
}

void* Ants::get_buffer(std::size_t type) {
    if(type == INT_BUFFER) {
        return static_cast<void*>(rainbow_board_);
    }
    else {
        return NULL;
    }
}

void Ants::set_buffer(void *new_buffer, std::size_t type) {
}

std::string Ants::get_name() {
    return "Ants";
}

/*
 * Other Standard Ruleset Functions
 */
void Ants::get_pixels(Buffer<Pixel<uint8_t>> *pixels) {
    if(rainbow_view_) {
        //rainbows_.age_to_pixels(rainbow_board_, pixels);
    }
    else {
        if(colony_pheromone_display_ > 0 &&
           colony_pheromone_display_ <= static_cast<int>(colonies_.size()))
        {
            auto colonies_it = colonies_.begin();
            std::advance(colonies_it, colony_pheromone_display_ - 1);
            (*colonies_it)->draw_pheromones((uint32_t*)pixels->get_data(false));
        }
        else {
            for(int j = 0; j < h_; j++) {
                for(int i = 0; i < w_; i++) {
                    pixels->set(i, j, {0, 0, 0, 0});
                }
            }
        }
        for(Colony *colony: colonies_) {
            colony->draw_self((uint32_t*)pixels->get_data(false));
        }
        for(Ant *ant: ants_) {
            int offset = ant->y * w_ + ant->x;
            uint32_t color = ant->colony->get_color();
            pixels->set(offset, uint32_to_pixel(color));
        }
        for(Food *food: foods_) {
            int offset = food->y * w_ + food->x;
            pixels->set(offset, uint32_to_pixel(0x0000FF00)); //Green food
        }
    }
}

std::string Ants::get_rule_string() {
    if(colony_pheromone_display_ > 0 &&
       colony_pheromone_display_ <= static_cast<int>(colonies_.size()))
    {
        auto colonies_it = colonies_.begin();
        std::advance(colonies_it, colony_pheromone_display_ - 1);
        return (*colonies_it)->get_dna_string();
    }
    return "";
}

void Ants::load_rule_string(std::string rules) {
    colonies_.push_back(new Colony(w_, h_,
                                   rand() % w_, rand() % h_,
                                   generate_color(),
                                   rules));
    colonies_.back()->add_ants(&ants_, 5);
}

void Ants::print_human_readable_rules() {
}

void Ants::start() {
    std::cout << "Starting Ants" << std::endl;
    Ruleset::start();

    InputManager::add_bool_toggler(&rainbow_view_, SDL_SCANCODE_T, false, false,
                                   "Ants", "Toggle rainbow view");

    ADD_FUNCTION_CALLER(&Ants::reset, SDL_SCANCODE_E, false, false,
                        "Ants", "Reset simulation");
    ADD_FUNCTION_CALLER_W_ARGS(&Ants::add_colony, VoidFunc, SDL_SCANCODE_R, false, false,
                               "Ants", "Add ant colony with random DNA", 5);

    InputManager::add_int_changer(&colony_pheromone_display_, SDL_SCANCODE_A,
                                  false, false, 0, INT_MAX,
                                  "Ants", "Pheromone display");
    InputManager::add_int_changer(&num_colonies_, SDL_SCANCODE_S,
                                  false, false, 0, INT_MAX,
                                  "Ants", "Minimum colonies");
    InputManager::add_int_changer(&color_speed_, SDL_SCANCODE_D,
                                  false, false, 0, INT_MAX,
                                  "Ants", "Color speed");
    InputManager::add_int_changer(&rainbow_train_len_, SDL_SCANCODE_Q,
                                  false, false, 0, INT_MAX,
                                  "Ants", "Trail length");
    InputManager::add_int_changer(&num_food_for_child_, SDL_SCANCODE_W,
                                  false, false, 0, INT_MAX,
                                  "Ants", "Num food to spawn child");
    rainbows_.start();
}

void Ants::stop() {
    Ruleset::stop();

    InputManager::remove_var_changer(SDL_SCANCODE_T, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_E, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_R, false, false);

    InputManager::remove_var_changer(SDL_SCANCODE_A, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_S, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_D, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_Q, false, false);
    InputManager::remove_var_changer(SDL_SCANCODE_W, false, false);
    rainbows_.stop();
}

void Ants::tick() {
    std::vector<Ant*> ant_to_remove;
    std::vector<Colony*> colonies_to_add;
    std::vector<Colony*> colonies_to_remove;

    cv::cuda::Stream cuda_stream;


    //Move every ant
    for(Ant *ant: ants_) {
        if(!ant->colony->move_ant(ant)) {
            ant_to_remove.push_back(ant);
        }
        else {
            rainbow_board_[ant->y * w_ + ant->x] = rainbow_train_len_;
        }
    }
    //Delete the ones that died
    for(Ant *ant: ant_to_remove) {
        ants_.remove(ant);
    }

    if(current_tick_ % color_speed_ == 0) {
        for(int j = 0; j < h_; j++) {
            for(int i = 0; i < w_; i++) {
                if(rainbow_board_[j * w_ + i] > 0) {
                    rainbow_board_[j * w_ + i]--;
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

            colony->queue_cuda_ops(cuda_stream);

        }
    }

    cuda_stream.waitForCompletion();


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
    memset(world_, 0, w_*h_*sizeof(world_[0]));

    for(Food *food: foods_) {
        int offset = food->y * w_ + food->x;
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
        int offset = ant->y * w_ + ant->x;
        bool overwrite_world = true;
        if(world_[offset].type == FoodType) {
            Food *f = static_cast<Food*>(world_[offset].ptr);
            if(!ant->has_food && f->bites_left > 0) {
                f->bites_left -= 1;
                if(f->bites_left == 0) {
                    delete f;
                    foods_.remove(f);
                    foods_.push_back(new Food(rand() % (w_ - 2) + 1,
                                              rand() % (h_ - 2) + 1,
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

/*
 * Ants Specific Functions
 */
void Ants::add_colony(int num_ants) {
    colonies_.push_back(new Colony(w_, h_,
                                   rand() % w_, rand() % h_,
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
    for(int i = 0; i < w_ * h_ / starting_food_density_; i++) {
        foods_.push_back(new Food(rand() % (w_ - 2) + 1,
                                  rand() % (h_ - 2) + 1,
                                  rand() % 50));
    }

    current_tick_ = 0;
    memset(rainbow_board_, 0, w_ * h_ * sizeof(int));
}

void Ants::restock_colonies(int num_ants) {
    while(static_cast<int>(colonies_.size()) < num_colonies_) {
        std::cout << "restocking" << std::endl;
        add_colony(num_ants);
    }
}

