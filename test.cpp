#include "layernorm.h"
#include <stdio.h>

int main() {
    Tensor x;
    int shape[2] = {1, 4};   // (N=1, C=4)
    tensor_init(&x, 2, shape);

    x.data[0] = 1.0f;
    x.data[1] = 2.0f;
    x.data[2] = 3.0f;
    x.data[3] = 4.0f;

    LayerNorm ln;
    layernorm_init(&ln, 4, 1e-5f);

    Tensor y;
    layernorm_forward(&ln, &x, &y);

    for (int i = 0; i < 4; i++)
        printf("y[%d] = %f\n", i, y.data[i]);

    tensor_free(&x);
    tensor_free(&y);
    layernorm_free(&ln);
}
