#ifndef _ANTS_FOOD_H
#define _ANTS_FOOD_H

struct Food {
    Food(int x_, int y_, int bites_left_)
        : x(x_)
        , y(y_)
        , bites_left(bites_left_)
    {}
    int x;
    int y;
    int bites_left;
};

#endif //_ANTS_FOOD_H
