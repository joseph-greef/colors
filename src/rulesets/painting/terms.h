
#ifndef _TERMS_H
#define _TERMS_H

#include <cmath>
#include <cstdarg>
#include <list>

#include "term.h"

class Addition : public Term {
private:
    std::list<Term*> terms_;
public:
    template<typename ... Terms>
    Addition(Terms ... terms) : Term() {
        Term* term_list[] = { static_cast<Term*>(terms)... };
        for(int i = 0; i < sizeof(term_list)/sizeof(term_list[0]); i++) {
            terms_.push_back(term_list[i]);
        }
    }
    ~Addition() {}
    double get_val() {
        double val = 0;
        for(Term *trm: terms_) {
            val += trm->get_val();
        }
        return val;
    }
};

class Multiplication : public Term {
private:
    std::list<Term*> terms_;
public:
    template<typename ... Terms>
    Multiplication(Terms ... terms) : Term() {
        Term* term_list[] = { static_cast<Term*>(terms)... };
        for(int i = 0; i < sizeof(term_list)/sizeof(term_list[0]); i++) {
            terms_.push_back(term_list[i]);
        }
    }
    ~Multiplication() {}
    double get_val() {
        double val = 1;
        for(Term *term: terms_) {
            val *= term->get_val();
        }
        return val;
    }
};

class Scalar : public Term {
private:
    double value_;
public:
    Scalar(double value) : Term(), value_(value) {}
    ~Scalar() {}
    double get_val() { return value_; }
    void increment(double val) { value_ += val; }
    void set_val(double val) { value_ = val; }
};

class Sin : public Term {
private:
    Term *amplitude_;
    Term *inner_;
public:
    Sin(Term *amplitude, Term *inner)
        : amplitude_(amplitude)
        , inner_(inner) {
    }
    double get_val() {
        return amplitude_->get_val() * std::sin(inner_->get_val());
    }
};

class Cos : public Sin {
public:
    Cos(Term *amplitude, Term *inner)
        : Sin(amplitude, new Addition(inner, new Scalar(M_PI/2)))
    {}
};

#endif //_TERMS_H
