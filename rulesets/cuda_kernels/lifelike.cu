
#include "lifelike.cuh"

#define VON_NEUMANN 0
#define MOORE 1

__device__ int get_num_alive_neighbors(int x, int y, int *board, 
                                       int width, int height) {

    int check_x, check_y, count = 0;
    for (int i = x - 1; i <= x + 1; i++) {
        for (int j = y - 1; j <= y + 1; j++) {
            if (j == y && i == x)
                continue;


            check_x = (i + width) % width;
            check_y = (j + height) % height;
            //and check the coordinate, if it's alive increase count
            if (board[check_y*width + check_x] > 0) {
                count++;
            }
        }
    }
    return count;

    /* A non-functional but maybe faster version waiting to be debuged.
    int top = (y + 1) % height;
    int bot = (y + height - 1) % height;
    int right = (x + 1) % width;
    int left = (x + width - 1) % width;

    int count = 0;
    count += !!board[top * width + left] + !!board[top * width + x] + !!board[top * width + right];
    count += !!board[y * width + left] +                          + !!board[y * width + right];
    count += !!board[bot * width + left] + !!board[bot * width + x] + !!board[bot * width + right];
        
    return count;
    */
}



__global__ void cuda_lifelike(int* board, int* board_buffer, bool *born, bool *stay_alive, int num_faders, int width, int height) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < height * width) {
        
        int j = index / width;
        int i = index % width;

        //get how many alive neighbors it has
        int neighbors = get_num_alive_neighbors(i, j, board, width, height);

#if reallylowdivergence
        //translate the divergent if statement into a longass boolean expression.
        int bornresult = (int) (born[neighbors]>0.5);
        int stayresult = (int) (stay_alive[neighbors]>0.5);
        int isdead = (int)(board[index] <= -num_faders);
        int isfader = (int) (board[index] > -num_faders && board[index] < 0);
        board_buffer[index] = !isfader * (isdead*(bornresult + !bornresult * (board[index] - 1)) + !isdead * (-(!stayresult) + stayresult * (board[index] + 1))) + isfader * (board[index] - 1);
#else       
        if (board[index] <= -num_faders) {
#if lowdivergence
            int result = (int) (born[neighbors]>0.5);
            board_buffer[index] = result + !result * (board[index] - 1);
#else
            if (born[neighbors]) {
                board_buffer[index] = 1;
            }
            else if (board[index] < 0)
                board_buffer[index] = board[index] - 1;
#endif /*lowdivergence*/
        }
        else if (board[index] > 0) {
#if lowdivergence
            int result = (int) (stay_alive[neighbors]>0.5);
            board_buffer[index] = -(!result) + result * (board[index] + 1);
#else
            if (stay_alive[neighbors])
                board_buffer[index] = board[index] + 1;
            else
                board_buffer[index] = -1;
#endif /*lowdivergence*/
            
        }
        else {
            board_buffer[index] = board[index] - 1;
        }

#endif /*reallylowdivergence*/


        index += blockDim.x * gridDim.x;
    }
}


void call_cuda_lifelike(int *board, int *board_buffer, bool *born, bool *stay_alive, int num_faders, int width, int height) {
    cuda_lifelike<<<512, 128>>>(board, board_buffer, born, stay_alive, num_faders, width, height);
}
