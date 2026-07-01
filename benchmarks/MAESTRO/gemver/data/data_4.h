#ifndef DATA_GEMVER_FUSED_H
#define DATA_GEMVER_FUSED_H

#include <stdint.h>

/* Dataset Size: 4 */
#define N 4
#define BETA 3
#define ALPHA 1

int A[16] = {
    4, -2, 3, 3, -5, 0, -2, 5, 4, 4, 5, 0, 2, 1, -5, -1
};

int u1[4] = {
    2, 3, 5, -4
};

int v1[4] = {
    5, 1, -3, -3
};

int u2[4] = {
    -4, -2, 5, 0
};

int v2[4] = {
    3, -4, 5, 3
};

int x[4] = {
    2, 3, -4, -5
};

int y[4] = {
    0, -1, -4, 0
};

int z[4] = {
    -1, 2, 1, -5
};

int w[4] = {
    -5, 4, -3, -1
};

int A_expected[16] = {
    2, 16, -23, -15, 4, 11, -21, -10, 44, -11, 15, 0, -18, -3, 7, 11
};

int x_expected[4] = {
    -539, 104, -120, 20
};

int w_expected[4] = {
    3041, 1312, -26663, 8769
};

#endif
