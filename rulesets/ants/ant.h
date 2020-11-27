#ifndef _ANTS_ANT_H
#define _ANTS_ANT_H

struct Ant {
    int x;
    int y;
    bool has_food;
    float steps_since_event;
    int colony_number;
    Colony *colony;
};

#endif //_ANTS_ANT_H
