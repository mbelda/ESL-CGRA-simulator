#include <stdio.h>

#define NI 20
#define NJ 30
#define NK 25

void mmul_majo(int inputX[NI*NK], int inputY[NK*NJ], int output[NI*NJ]){
  int i,j,k;
  int sum;
  for(i = 0; i < NI; i ++) {
        for(j = 0; j < NJ; j ++) {
            sum = 0;
            for(k = 0; k < NK; k++) {
                sum += inputX[i * NK + k] * inputY[k * NJ + j];
            }
            output[i * NJ + j] = sum;
        }
    }
}