
#include "initializer.h"


/*
//create the board initializer with some arbitrary starting values
Initializer::Initializer(Board *b) : e2(time(NULL)){
    board = b;
    density = 50;
    num_gliders = 4;
}
*/
Initializer::Initializer(int width, int height)
    : height_(height)
    , width_(width)
{
}

//nothing to deconstruct, why is this even here?
Initializer::~Initializer() {

}
//clears the board. If changing_background is true sets everything to -1
//so it will age, otherwise sets it to 0 so it won't
void Initializer::clear_board(int *board) {
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = 0;
        }
    }
}

//randomly initializes the board with density percent alive cells
void Initializer::init_board(int *board, int density) {
    for (int i = 0; i < width_; i++) {
        for (int j = 0; j < height_; j++) {
            board[j* width_ + i] = (rand() % 100 < density ? 1 : -1);
        }
    }
}

//clears the board and draws a dot in the center with side length density/10
void Initializer::init_center_square(int *board, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            board[j * width_ + i] = 1;
        }
    }
}

void Initializer::init_center_diamond(int *board, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            if(abs(i - width_ / 2)+abs(j - height_ / 2) < radius) {
                board[j * width_ + i] = 1;
            }
        }
    }
}


void Initializer::init_center_cross(int *board, int line_width, int radius) {
    clear_board(board);
    for (int i = width_ / 2 - line_width; i < width_ / 2 + line_width; i++) {
        for (int j = height_ /  2 - radius; j < height_ / 2 + radius; j++) {
            board[j * width_ + i] = 1;
        }
    }
    for (int i = width_ / 2 - radius; i < width_ / 2 + radius; i++) {
        for (int j = height_ /  2 - line_width; j < height_ / 2 + line_width; j++) {
            board[j * width_ + i] = 1;
        }
    }
}

#if 0
void Initializer::init_symm() {

}

//create a board with valid hodge numbers
void Initializer::init_hodge_board(int n) {
    //int count = 0;
    int *b = board->get_board();
    for (int i = 0; i < board->get_cell_width(); i++) {
        for (int j = 0; j < board->get_cell_height(); j++) {
            b[j*board->get_cell_width() + i] = (rand() % n);
            //b[j*board->get_cell_width() + i] = count;
            //count = (count + 1) % n;
        }
    }
    board->send_board_to_GPU();
}

//puts num_gliders/4 dots of side length density/10 in each of the
//four quadrants. There is vertical and horizontal symmetry
void Initializer::init_quadrants() {
    int *b = board->get_board();
    clear_board(b);
    int x, y;
    for (int i = 0; i < num_gliders / 4; i++) {
        x = rand() % (board->get_cell_width() / 2 - density / 10);
        y = rand() % (board->get_cell_height() / 2 - density / 10);

        //u++ <=> u = u + 1
        for (int u = 0; u < density / 10; u++) {
            for (int v = 0; v < density / 10; v++) {
                //top left
                b[(y + u)*board->get_cell_width() + (x + v)] = 1;
                //top right quadrant
                b[(y + u)*board->get_cell_width() + (board->get_cell_width() - x - v)] = 1;
                //bottom left
                b[(board->get_cell_height() - y - u)*board->get_cell_width() + (x + v)] = 1;
                //bottom right
                b[(board->get_cell_height() - y - u)*board->get_cell_width() + (board->get_cell_width() - x - v)] = 1;
            }
        }
    }
    board->send_board_to_GPU();
}


//initialize the floating point board
void Initializer::init_smooth_life() {
    float *b = board->get_board_float();
    int *b2 = board->get_board();
    clear_board(b2);
    for (int i = 0; i < board->get_cell_width(); i++) {
        for (int j = 0; j < board->get_cell_height(); j++) {
            b[j*board->get_cell_width() + i] = (rand() % 100 < density ? 1 : 0);
        }
    }
    board->send_board_to_GPU();
}

//this function clears the board then makes num_glider gliders in random
//locations and orientations. Makes sure to keep them away from the edge so
//we don't get seg faults
void Initializer::init_gliders() {
    int *b = board->get_board();
    clear_board(b);
    for (int i = 0; i < num_gliders; i++) {
        make_glider(rand() % (board->get_cell_width() - 10) + 5, rand() % (board->get_cell_height() - 10) + 5, rand() % 4);
    }
    board->send_board_to_GPU();
}


//creates a square shell of a random size in the center of the screen
void Initializer::init_square_shell() {
    int *b = board->get_board();
    clear_board(b);
    int center_x = board->get_cell_width() / 2;
    int center_y = board->get_cell_height() / 2;
    int side_length = rand() % (board->get_cell_height() / 2 - 1) + 1;
    for (int i = center_x - side_length; i < center_x + side_length; i++) {
        for (int j = center_y - side_length; j < center_y + side_length; j++) {
            b[j*board->get_cell_width() + i] = 1;

        }
    }
    side_length-=4;
    for (int i = center_x - side_length; i < center_x + side_length; i++) {
        for (int j = center_y - side_length; j < center_y + side_length; j++) {
            b[j*board->get_cell_width() + i] = changing_background ? -1 : 0;

        }
    }
    board->send_board_to_GPU();
}

//creates a circular cell at the center of the screen
void Initializer::init_circle_shell() {
    int *b = board->get_board();
    clear_board(b);
    int r = rand() % (board->get_cell_height() / 2 - 1) + 1;
    int center_x = board->get_cell_width() / 2;
    int center_y = board->get_cell_height() / 2;
    int* circle_coords = (int*)malloc(sizeof(int) * r * r * 4 * 2);
    get_circle(center_x, center_y, r, circle_coords);
    int index = 0;
    int i, j;
    while (circle_coords[index] != -1) {
        i = circle_coords[index];
        index++;
        j = circle_coords[index];
        index++;

        b[j*board->get_cell_width() + i] = 1;
    }
    r-=4;
    index = 0;
    get_circle(center_x, center_y, r, circle_coords);
    while (circle_coords[index] != -1) {
        i = circle_coords[index];
        index++;
        j = circle_coords[index];
        index++;

        b[j*board->get_cell_width() + i] = changing_background ? -1 : 0;
    }
    free(circle_coords);
    board->send_board_to_GPU();
}

//creates an arbitrary polygonal shell at the center of the screen
void Initializer::init_polygon_shell() {
    int *b = board->get_board();
    clear_board(b);
    int mag = 20 + rand() % 50;// rand() % (board->get_cell_height()/8 -1) + 1;
    int* coords = (int*)malloc(2 * sizeof(int) *board->get_cell_height() * board->get_cell_width());
    get_polygon(board->get_cell_width() / 2, board->get_cell_height() / 2, mag, num_gliders, 0, coords);
    int index = 0;
    int i, j;
    while (coords[index] != -1) {
        i = coords[index];
        index++;
        j = coords[index];
        index++;

        b[j*board->get_cell_width() + i] = 1;
    }
    board->send_board_to_GPU();
    return;



}


//puts a circle at a random spot on the screen
void Initializer::init_circle() {
    int *b = board->get_board();
    int r = rand() % 30 + 3;
    int* circle_coords = (int*)malloc(sizeof(int) * r * r * 4 * 2);
    get_circle(rand() % board->get_cell_width(), rand() % board->get_cell_height() - 10, r, circle_coords);
    int index = 0;
    int i, j;
    while (circle_coords[index] != -1) {
        i = circle_coords[index];
        index++;
        j = circle_coords[index];
        index++;

        b[j*board->get_cell_width() + i] = 1;
    }
    free(circle_coords);
    board->send_board_to_GPU();
}

//clears the board. If changing_background is true sets everything to -1
//so it will age, otherwise sets it to 0 so it won't
void Initializer::clear_board(int *b) {
    int deadnum = -board->get_faders() - 1;
    for (int i = 0; i < board->get_cell_width(); i++) {
        for (int j = 0; j < board->get_cell_height(); j++) {
            b[j*board->get_cell_width() + i] = changing_background ? deadnum : 0;
        }
    }
    board->send_board_to_GPU();
}


//put a dot in the center of the bottom row for the 1d automata
void Initializer::init_1D_board() {
    int *b = board->get_board();
    clear_board(b);
    int j = board->get_cell_height() - 1;
    int i = board->get_cell_width() / 2;
    b[j*board->get_cell_width() + i] = 1;
    board->send_board_to_GPU();
}


//this function creates a glider on board at x,y in one of 4 orientations.
//it just manually sets each required alive cell. Assumes that all cells are
//dead in the required area. Doesn't account for the edge of the board
void Initializer::make_glider(int x, int y, int orientation) {
    int *b = board->get_board();
    if (orientation == 0) {
        b[(y + 0)*board->get_cell_width() + (x + 1)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 2)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 1)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 2)] = 1;
    }
    else if (orientation == 1) {
        b[(y + 0)*board->get_cell_width() + (x + 1)] = 1;
        b[(y + 0)*board->get_cell_width() + (x + 2)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 2)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 2)] = 1;
    }
    else if (orientation == 2) {
        b[(y + 0)*board->get_cell_width() + (x + 1)] = 1;
        b[(y + 0)*board->get_cell_width() + (x + 2)] = 1;
        b[(y + 0)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 1)] = 1;
    }
    else if (orientation == 3) {
        b[(y + 0)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 1)*board->get_cell_width() + (x + 2)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 0)] = 1;
        b[(y + 2)*board->get_cell_width() + (x + 1)] = 1;
    }
}


void Initializer::set_density(int new_density) {
    density = new_density;
}

void Initializer::modify_gliders(int factor) {
    num_gliders += factor;
    if (num_gliders < 2)
        num_gliders = 2;
}


//finds a polygon of num_gliders sides, with some irregularity.
void Initializer::get_polygon(int center_x, int center_y, int mag, int dim, float irreg, int* points) {
    int index = 0;


    std::uniform_real_distribution<> udist(-PI, PI);
    std::uniform_real_distribution<> udist2((1 - irreg) * 2 * PI / dim, (1 + irreg) * 2 * PI / dim);


    float **v = (float**)malloc(sizeof(float*)*dim);

    float x_sum = 0, y_sum = 0;
    float theta = udist(e2);

    int k;
    for (k = 0; k < dim - 1; k++) {
        theta += udist2(e2);
        v[k] = (float*)malloc(sizeof(float) * 2);
        v[k][0] = std::cos(theta);
        v[k][1] = std::sin(theta);
        x_sum += v[k][0];
        y_sum += v[k][1];
    }
    v[k] = (float*)malloc(sizeof(float) * 2);
    v[k][0] = -1 * x_sum;
    v[k][1] = -1 * y_sum;

    float norm_factor = v[k][0] * v[k][0] + v[k][1] * v[k][1];
    v[k][0] = v[k][0] / norm_factor;
    v[k][1] = v[k][1] / norm_factor;




    for (int i = 0; i < board->get_cell_width(); i++) {
        for (int j = 0; j < board->get_cell_height(); j++) {
            float x = i - center_x;
            float y = j - center_y;

            bool inside1 = true;
            bool inside2 = false;
            for (int k = 0; k < dim; k++) {
                inside1 = inside1 && (x*v[k][0] + y*v[k][1]) < mag;
                inside2 = inside2 || (x*v[k][0] + y*v[k][1]) > mag - 5;

            }
            if (inside1 && inside2) {
                points[index] = i;
                index++;
                points[index] = j;
                index++;
            }
            /*if(inside2) {
            points2[index] = i;
            index++;
            points2[index] = j;
            index++;
            }*/


        }
    }
    points[index] = -1;
    for (int k = 0; k < dim; k++) {
        free(v[k]);
    }
    free(v);
}



//gets a circle and puts its points in points
void Initializer::get_circle(int x, int y, int r, int* points) {
    int index = 0;
    int test_x, test_y;
    int rsquared = r*r;
    for (int i = -r; i <= r; i++) {
        test_x = i + x;
        if (test_x < 0)
            test_x += board->get_cell_width();
        else if (test_x >= board->get_cell_width())
            test_x -= board->get_cell_width();

        //int yspan = r*sin(acos(-i/r));
        //for(int j = -yspan; j <= yspan; j++) {
        for (int j = -r; j <= r; j++) {
            test_y = j + y;
            if (test_y < 0)
                test_y += board->get_cell_height();
            else if (test_y >= board->get_cell_height())
                test_y -= board->get_cell_height();


            if (i * i + j * j <= rsquared) {
                points[index] = test_x;
                index++;
                points[index] = test_y;
                index++;
            }

        }
    }
    points[index] = -1;
}

#endif

