// softmax.h
// Numerically stable softmax along the LAST dimension.

#pragma once

#include "tensor.h"
#include <math.h>
#include <stdio.h>

static inline void softmax_set_distributed(int rank, int world_size) {
    (void)rank;
    (void)world_size;
}

// Context for the softmax operation
typedef struct {
    Tensor* input;
    Tensor* output;
} SoftmaxContext;

// Backward function for 2D softmax
static inline void softmax_backward_2d(Tensor* t) {
    SoftmaxContext* ctx = (SoftmaxContext*)t->_ctx;
    Tensor* x = ctx->input;
    Tensor* y = ctx->output;
    
    int N = x->shape[0];
    int C = x->shape[1];

    for (int n = 0; n < N; ++n) {
        // Compute dot product of y and grad_y for this row
        float dot_product = 0.0f;
        for (int c = 0; c < C; ++c) {
            dot_product += tensor_get2(y, n, c) * y->grad[tensor_index2(y, n, c)];
        }
        
        // Compute grad_x for this row
        for (int c = 0; c < C; ++c) {
            float y_val = tensor_get2(y, n, c);
            float grad_y_val = y->grad[tensor_index2(y, n, c)];
            x->grad[tensor_index2(x, n, c)] += y_val * (grad_y_val - dot_product);
        }
    }
    free(ctx);
}

// Backward function for 3D softmax
static inline void softmax_backward_3d(Tensor* t) {
    SoftmaxContext* ctx = (SoftmaxContext*)t->_ctx;
    Tensor* x = ctx->input;
    Tensor* y = ctx->output;

    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Compute dot product of y and grad_y for this slice
            float dot_product = 0.0f;
            for (int c = 0; c < C; ++c) {
                dot_product += tensor_get3(y, b, t, c) * y->grad[tensor_index3(y, b, t, c)];
            }
            
            // Compute grad_x for this slice
            for (int c = 0; c < C; ++c) {
                float y_val = tensor_get3(y, b, t, c);
                float grad_y_val = y->grad[tensor_index3(y, b, t, c)];
                x->grad[tensor_index3(x, b, t, c)] += y_val * (grad_y_val - dot_product);
            }
        }
    }
    free(ctx);
}


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
    
    // Create context for autograd
    SoftmaxContext* ctx = (SoftmaxContext*)malloc(sizeof(SoftmaxContext));
    ctx->input = (Tensor*)x;
    ctx->output = y;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = softmax_backward_2d;
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
    
    // Create context for autograd
    SoftmaxContext* ctx = (SoftmaxContext*)malloc(sizeof(SoftmaxContext));
    ctx->input = (Tensor*)x;
    ctx->output = y;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = softmax_backward_3d;
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
