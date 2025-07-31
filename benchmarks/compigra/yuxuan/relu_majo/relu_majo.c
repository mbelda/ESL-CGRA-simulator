#include <stdio.h>

#define N 128
#define M 128

// void relu_majo(int input[N*M], int max_iter_ptr[1]){
//     int max_iter = max_iter_ptr[0] >> 2;
//     for(int i=0; i < max_iter; i++){
//         int val0 = input[4*i];
//         int val1 = input[4*i+1];
//         int val2 = input[4*i+2];
//         int val3 = input[4*i+3];
//         if (val0 < 0){
//             val0 = 0;
//         }
//         if (val1 < 0){
//             val1 = 0;
//         }
//         if (val2 < 0){
//             val2 = 0;
//         }
//         if (val3 < 0){
//             val3 = 0;
//         }
//         input[4*i] = val0;
//         input[4*i+1] = val1;
//         input[4*i+2] = val2;
//         input[4*i+3] = val3;
//     }
// }

void relu_majo(int input[N*M], int max_iter_ptr[1]){
    int max_iter = max_iter_ptr[0];
    for(int i=0; i < max_iter; i++){
        if (input[i] < 0){
            input[i] = 0;
        }
    }
}