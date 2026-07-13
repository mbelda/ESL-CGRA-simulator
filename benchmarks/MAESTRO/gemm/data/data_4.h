#ifndef DATA_H
#define DATA_H

#include <stdint.h>

/* Dataset Dimensions: NI=4, NK=4, NJ=4 */
#define NI 4
#define NK 4
#define NJ 4

#define ALPHA 1
#define BETA 1

int A[16] = {
    5, 3, 4, -2, 3, 3, -5, 0, -2, 5, 4, 4, 5, 0, 2, 1
};

int B[16] = {
    -5, -1, 2, 3, 5, -4, 5, 1, -3, -3, -4, -2, 5, 0, 3, -4
};

int C[16] = {
    5, 3, 2, 3, -4, -5, 0, -1, -4, 0, -1, 2, 1, -5, -5, 4
};

int C_expected[16] = {
    -27, -26, 5, 21, 11, -5, 41, 21, 39, -30, 16, -23, -25, -16, 0, 11
};

#endif
