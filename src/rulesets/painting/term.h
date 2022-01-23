
#ifndef _TERM_H
#define _TERM_H

class Term {
public:
    virtual ~Term();
    virtual double get_val() = 0;
};

#endif //_TERM_H
