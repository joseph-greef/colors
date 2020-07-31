
#include "lifelike.h"
#include <iostream>

LifeLike::LifeLike(int width, int height)
    : Ruleset(width, height)
    , _alive_color_scheme(0)
    , _alive_offset(128)
    , _dead_color_scheme(0)
    , _dead_offset(0)
    , _initializer(width, height)
    , _num_faders(0)
    , _rainbows()
{
    bool born_tmp[9] = {0, 0, 0, 0, 1, 1 ,1 ,1, 1};
    bool stay_alive_tmp[9] = {1, 0, 0, 1, 1, 1 ,1 ,1, 0};

    memcpy(_born, born_tmp, sizeof(_born));
    memcpy(_stay_alive, stay_alive_tmp, sizeof(_stay_alive));

    _board = new int[width*height];
    _board_buffer = new int[width*height];

    _initializer.init_board(_board);
    for(int i = 0; i < 9; i++) {
        std::cout << _born[i] << " " <<  _stay_alive[i] << std::endl;
    }
}

LifeLike::~LifeLike() {
    delete _board;
    delete _board_buffer;
}

void LifeLike::get_pixels(uint32_t *pixels) {
    Rainbows::age_to_pixels(_board, pixels,
                            _alive_color_scheme, _alive_offset,
                            _dead_color_scheme, _dead_offset,
                            _width, _height);
}

void LifeLike::tick() {
    for(int j = 0; j < _height; j++) {
        for(int i = 0; i < _width; i++) {
            //get how many alive neighbors it has
            int neighbors = Ruleset::get_num_alive_neighbors(_board, i, j, 1,
                                                             Moore);
            int offset = j * _width + i;
            //alive
            if (_board[offset] > 0) {
                if(_stay_alive[neighbors])
                    //then age it
                    _board_buffer[offset] = _board[offset] + 1;
                else
                    //otherwise kill it
                    _board_buffer[offset] = -1;

            }
            //dead
            else if(_board[offset] <= -_num_faders) {
                if(_born[neighbors])
                    _board_buffer[offset] = 1;
                //Don't age the cell if it's value is 0 to enable only seeing 
                //live cells
                else if(_board[offset] == 0)
                    _board_buffer[offset] = 0;
                else
                    _board_buffer[offset] = _board[offset] - 1;
            }
            //cell in refractory period 
            else {
                _board_buffer[offset] = _board[offset] - 1;
            }
        }
    }

    {
        int *tmp = _board_buffer;
        _board_buffer = _board;
        _board = tmp;
    }

}

