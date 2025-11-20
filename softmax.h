// softmax.h
// Numerically stable softmax along the LAST dimension.

#pragma once

#include "tensor.h"
#include <math.h>
#include <stdio.h>

/*
    Softmax for a (N, C) tensor over last dimension C.
*/
static inline void softmax_forward_2d(const Tensor* x, Tensor* y) {
    int N = x->shape[0];
    int C = x->shape[1];

    int y_shape[2] = {N, C};
    tensor_init(y, 2, y_shape);

    for (int n = 0; n < N; ++n) {
        // 1. find max for stability
        float maxv = -1e30f;
        for (int c = 0; c < C; ++c) {
            float v = tensor_get2(x, n, c);
            if (v > maxv) maxv = v;
        }

        // 2. compute exp(x - maxv) and sum
        float sum = 0.0f;
        float* temp = (float*)malloc(C * sizeof(float)); // temporary buffer
        for (int c = 0; c < C; ++c) {
            float ex = expf(tensor_get2(x, n, c) - maxv);
            temp[c] = ex;
            sum += ex;
        }

        // 3. normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            tensor_set2(y, n, c, temp[c] * inv_sum);
        }

        free(temp);
    }
}


/*
    Softmax for a (B, T, C) tensor over last dimension C.
*/
static inline void softmax_forward_3d(const Tensor* x, Tensor* y) {
    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    int y_shape[3] = {B, T, C};
    tensor_init(y, 3, y_shape);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {

            // 1. find max
            float maxv = -1e30f;
            for (int c = 0; c < C; ++c) {
                float v = tensor_get3(x, b, t, c);
                if (v > maxv) maxv = v;
            }

            // 2. compute exps
            float sum = 0.0f;
            float* temp = (float*)malloc(C * sizeof(float));
            for (int c = 0; c < C; ++c) {
                float ex = expf(tensor_get3(x, b, t, c) - maxv);
                temp[c] = ex;
                sum += ex;
            }

            float inv_sum = 1.0f / sum;

            // 3. normalize
            for (int c = 0; c < C; ++c) {
                tensor_set3(y, b, t, c, temp[c] * inv_sum);
            }

            free(temp);
        }
    }
}


/*
    Wrapper that dispatches based on ndim
*/
static inline void softmax_forward(const Tensor* x, Tensor* y) {
    if (x->ndim == 2) {
        softmax_forward_2d(x, y);
    } else if (x->ndim == 3) {
        softmax_forward_3d(x, y);
    } else {
        printf("softmax_forward: ERROR: x->ndim must be 2 or 3\n");
    }
}

