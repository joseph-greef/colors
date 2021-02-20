
#include <cmath>
#include <iostream>

#include "transformation.h"


Transformation::Transformation(int width, int height)
    : height_(height)
    , width_(width)
    , e2_(rd_())
    , dist_full_(-1, 1)
    , dist_positive_(0, 1)
{
}

Transformation::~Transformation() {
}

