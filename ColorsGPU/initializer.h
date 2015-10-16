#include "board.h"

#include "info.h"

#include <random>

class Initializer {
private:
    
    Board *board;
    bool changing_background;
    int density;
    int num_gliders;
    std::mt19937 e2;


    void get_circle(int x, int y, int r, int* points);
    void get_polygon(int x, int y, int mag, int dim, float irreg, int* points);


    void update_board_normal();
    void update_board_smooth();
    void update_board_LtL();

    
public:
    Initializer(Board *b);
    ~Initializer();

    void init_board();
    void init_hodge_board(int n);
    void init_quadrants();
    void init_center_dot();
    void init_gliders();
    void init_circle();
    void init_symm();
    void init_smooth_life();
    void init_square_shell();
    void init_circle_shell();
    void init_polygon_shell();

    void init_1D_board();

    void clear_board(int* b);

    void make_glider(int x, int y, int orientation);

    void set_density(int new_density);

    void modify_gliders(int factor);

};