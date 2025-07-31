#include <stdio.h>

#define IM_HEIGHT 128
#define IM_WIDTH 128

void conv2d_majo(int image[IM_HEIGHT*IM_WIDTH], 
                 int filter[9], int output[IM_HEIGHT*IM_WIDTH]){
    int x,y,kx,ky;
    for (y = 1; y < IM_HEIGHT - 1; y++) {
        for (x = 1; x < IM_WIDTH - 1; x++) {
            int sum = 0;
            for (ky = -1; ky <= 1; ky++) {
                int partial_sum = 0;
                for (kx = -1; kx <= 1; kx++) {
                    int in_x = x + kx;
                    int in_y = y + ky;

                    int input_val = image[in_y * IM_WIDTH + in_x];
                    int filter_val = filter[(ky + 1) * 3 + (kx + 1)];
                    partial_sum += input_val * filter_val;
                }
                sum += partial_sum;
            }
            output[y * IM_WIDTH + x] = sum;
        }
    }
}