

#include "stroke.h"
#include <iostream>

Stroke::Stroke() {
    time_ = new Scalar(0);
    x_term_ = new Addition(new Scalar(300),
                           new Sin(new Scalar(30), new Multiplication(new Scalar(1/180),
                                                                      time_)),
                           new Multiplication(new Scalar(0.1), time_));
    y_term_ = new Addition(new Scalar(300),
                           new Sin(new Scalar(30), new Multiplication(new Scalar(1/180),
                                                                      time_)),
                           new Multiplication(new Scalar(0.1), time_));
}

Stroke::~Stroke() {
    //TODO: Implement deconstruction of terms
}

void Stroke::mark(Buffer<int> *board, double time) {
    board->set(x_term_->get_val(), y_term_->get_val(), int(time_->get_val()));
    time_->increment(1);
    std::cout << int(time) << std::endl;
}
