
#ifndef _STROKE_H
#define _STROKE_H

#include "terms.h"
#include "buffer.cuh"

#include <list>

class Stroke {
private:
    Scalar *time_;
    Term *x_term_;
    Term *y_term_;

public:
    Stroke();
    ~Stroke();

    void mark(Buffer<int> *board, double time);

};

#endif //_STROKE_H
