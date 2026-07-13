#ifndef DATA_H
#define DATA_H

#include <stdint.h>

/* Dataset Dimensions: NI=4, NK=3, NJ=4 */
#define NI 4
#define NK 3
#define NJ 4

#define ALPHA 1
#define BETA 1

int A[12] = {
    5, 3, 4, -2, 3, 3, -5, 0, -2, 5, 4, 4
};

int B[12] = {
    5, 0, 2, 1, -5, -1, 2, 3, 5, -4, 5, 1
};

int C[16] = {
    -3, -3, -4, -2, 5, 0, 3, -4, 5, 3, 2, 3, -4, -5, 0, -1
};

int C_expected[16] = {
    27, -22, 32, 16, -5, -15, 20, 6, -30, 11, -18, -4, 21, -25, 38, 20
};

#endif
