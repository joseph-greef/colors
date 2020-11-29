#ifndef _ANTS_ANT_H
#define _ANTS_ANT_H

struct Ant {
    Ant(int x_, int y_, int colony_number_, Colony *colony_)  
        : x(x_)
        , y(y_)
        , enemy_seen(false)
        , has_food(false)
        , steps_since_event(0)
        , total_steps(0)
        , colony_number(colony_number_)
        , colony(colony_) 
    {}

    int x;
    int y;
    bool enemy_seen;
    bool has_food;
    float steps_since_event;
    int total_steps;
    int colony_number;
    Colony *colony;
};

#endif //_ANTS_ANT_H
