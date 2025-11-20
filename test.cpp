#include <stdio.h>
#include "tensor.h"
#include "softmax.h"

int main() {
    // Create a simple 2D tensor: shape (1, 4)
    Tensor x;
    int shape[2] = {1, 4};
    tensor_init(&x, 2, shape);

    // Values: [-1, 0, 1, 2]
    x.data[0] = -1.0f;
    x.data[1] =  0.0f;
    x.data[2] =  1.0f;
    x.data[3] =  2.0f;

    Tensor y;
    softmax_forward(&x, &y);

    printf("Softmax output:\n");
    for (int i = 0; i < 4; i++) {
        printf("y[%d] = %f\n", i, y.data[i]);
    }

    tensor_free(&x);
    tensor_free(&y);

    // Test 3D
    Tensor x3;
    int shape3[3] = {1, 2, 3}; // (B=1,T=2,C=3)
    tensor_init(&x3, 3, shape3);

    // Fill with:
    // t=0: [1,2,3]
    // t=1: [4,5,6]
    x3.data[0] = 1.0f;
    x3.data[1] = 2.0f;
    x3.data[2] = 3.0f;
    x3.data[3] = 4.0f;
    x3.data[4] = 5.0f;
    x3.data[5] = 6.0f;

    Tensor y3;
    softmax_forward(&x3, &y3);

    printf("\nSoftmax 3D output:\n");
    for (int i = 0; i < 6; i++) {
        printf("y3[%d] = %f\n", i, y3.data[i]);
    }

    tensor_free(&x3);
    tensor_free(&y3);
    return 0;
}
