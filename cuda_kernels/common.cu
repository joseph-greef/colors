
#include "common.cuh"

__device__ int get_num_alive_neighbors(int x, int y, int neighborhood_type, int range, int *board, int width, int height) {
    int check_x, check_y, count = 0;

    if (neighborhood_type == VON_NEUMANN) {
        for (int i = x - range; i <= x + range; i++) {
            for (int j = y - range; j <= y + range; j++) {
                if (j == y && i == x)
                    continue;
                if (abs(i - x) + abs(j - y) <= range) {


                    check_x = (i + width) % width;
                    check_y = (j + height) % height;
                    //and check the coordinate, if it's alive increase count
                    if (board[check_y*width + check_x] > 0)
                        count++;
                }
            }
        }
    }
    else {
        for (int i = x - range; i <= x + range; i++) {
            for (int j = y - range; j <= y + range; j++) {
                if (j == y && i == x)
                    continue;


                check_x = (i + width) % width;
                check_y = (j + height) % height;
                //and check the coordinate, if it's alive increase count
                if (board[check_y*width + check_x] > 0)
                    count++;
            }
        }
    }
    return count;
}



