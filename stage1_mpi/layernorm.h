// layernorm.h
// Implements LayerNorm over the LAST dimension, like PyTorch's nn.LayerNorm

#pragma once

#include "tensor.h"
#include <math.h>
#include <stdio.h>

static inline void layernorm_set_distributed(int rank, int world_size) {
    (void)rank;
    (void)world_size;
}

typedef struct {
    int normalized_dim;  // equals n_embd
    float eps;

    Tensor gamma;  // scale parameter (n_embd)
    Tensor beta;   // bias  parameter (n_embd)
} LayerNorm;

// Context for the layernorm operation
typedef struct {
    LayerNorm* ln_layer;
    Tensor* input;
    Tensor* normalized;
} LayerNormContext;


/*
  Initialize LayerNorm with embedding dimension n_embd.
  gamma initialized to 1
  beta  initialized to 0
*/
static inline void layernorm_init(LayerNorm* ln, int n_embd, float eps) {
    ln->normalized_dim = n_embd;
    ln->eps = eps;

    int shape1[1];
    shape1[0] = n_embd;

    tensor_init(&ln->gamma, 1, shape1);  // (n_embd)
    tensor_init(&ln->beta , 1, shape1);  // (n_embd)

    // gamma = 1.0, beta = 0.0
    for (int i = 0; i < n_embd; ++i) {
        tensor_set1(&ln->gamma, i, 1.0f);
        tensor_set1(&ln->beta , i, 0.0f);
    }
}


/*
  Free LayerNorm parameters.
*/
static inline void layernorm_free(LayerNorm* ln) {
    tensor_free(&ln->gamma);
    tensor_free(&ln->beta);
}

static inline void layernorm_collect_params(LayerNorm* ln, TensorPtrArray* list) {
    tensor_ptr_array_push(list, &ln->gamma);
    tensor_ptr_array_push(list, &ln->beta);
}

// Backward function for 2D layernorm
static inline void layernorm_backward_2d(Tensor* t) {
    LayerNormContext* ctx = (LayerNormContext*)t->_ctx;
    LayerNorm* ln = ctx->ln_layer;
    Tensor* x = ctx->input;
    Tensor* y = t;
    
    int N = x->shape[0];
    int C = ln->normalized_dim;
    
    for (int n = 0; n < N; ++n) {
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) mean += tensor_get2(x, n, c);
        mean /= C;

        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float diff = tensor_get2(x, n, c) - mean;
            var += diff * diff;
        }
        var /= C;
        float inv_std = 1.0f / sqrtf(var + ln->eps);

        // Gradients for gamma and beta
        for (int c = 0; c < C; ++c) {
            ln->gamma.grad[c] += y->grad[tensor_index2(y, n, c)] * (tensor_get2(x, n, c) - mean) * inv_std;
            ln->beta.grad[c] += y->grad[tensor_index2(y, n, c)];
        }

        // Gradient for input x
        float dnorm_dx_sum = 0;
        float dnorm_dx_mul_x_minus_mean_sum = 0;
        for (int c = 0; c < C; c++) {
            dnorm_dx_sum += y->grad[tensor_index2(y, n, c)] * tensor_get1(&ln->gamma, c);
            dnorm_dx_mul_x_minus_mean_sum += y->grad[tensor_index2(y, n, c)] * tensor_get1(&ln->gamma, c) * (tensor_get2(x, n, c) - mean);
        }

        for (int c = 0; c < C; c++) {
            float dx_hat = y->grad[tensor_index2(y, n, c)] * tensor_get1(&ln->gamma, c);
            float term1 = C * dx_hat;
            float term2 = dnorm_dx_sum;
            float term3 = (tensor_get2(x, n, c) - mean) * dnorm_dx_mul_x_minus_mean_sum * inv_std * inv_std;
            x->grad[tensor_index2(x, n, c)] += (term1 - term2 - term3) * inv_std / C;
        }
    }
    
    free(ctx);
}

// Backward function for 3D layernorm
static inline void layernorm_backward_3d(Tensor* t) {
    LayerNormContext* ctx = (LayerNormContext*)t->_ctx;
    LayerNorm* ln = ctx->ln_layer;
    Tensor* x = ctx->input;
    Tensor* y = t;

    int B = x->shape[0];
    int T = x->shape[1];
    int C = ln->normalized_dim;

    for (int b = 0; b < B; ++b) {
        for (int t_ = 0; t_ < T; ++t_) {
            float mean = 0.0f;
            for (int c = 0; c < C; ++c) mean += tensor_get3(x, b, t_, c);
            mean /= C;

            float var = 0.0f;
            for (int c = 0; c < C; ++c) {
                float diff = tensor_get3(x, b, t_, c) - mean;
                var += diff * diff;
            }
            var /= C;
            float inv_std = 1.0f / sqrtf(var + ln->eps);

            // Gradients for gamma and beta
            for (int c = 0; c < C; ++c) {
                ln->gamma.grad[c] += y->grad[tensor_index3(y, b, t_, c)] * (tensor_get3(x, b, t_, c) - mean) * inv_std;
                ln->beta.grad[c] += y->grad[tensor_index3(y, b, t_, c)];
            }

            // Gradient for input x
            float dnorm_dx_sum = 0;
            float dnorm_dx_mul_x_minus_mean_sum = 0;
            for (int c = 0; c < C; c++) {
                dnorm_dx_sum += y->grad[tensor_index3(y, b, t_, c)] * tensor_get1(&ln->gamma, c);
                dnorm_dx_mul_x_minus_mean_sum += y->grad[tensor_index3(y, b, t_, c)] * tensor_get1(&ln->gamma, c) * (tensor_get3(x, b, t_, c) - mean);
            }

            for (int c = 0; c < C; c++) {
                float dx_hat = y->grad[tensor_index3(y, b, t_, c)] * tensor_get1(&ln->gamma, c);
                float term1 = C * dx_hat;
                float term2 = dnorm_dx_sum;
                float term3 = (tensor_get3(x, b, t_, c) - mean) * dnorm_dx_mul_x_minus_mean_sum * inv_std * inv_std;
                x->grad[tensor_index3(x, b, t_, c)] += (term1 - term2 - term3) * inv_std / C;
            }
        }
    }
    
    free(ctx);
}


/*
  Apply LayerNorm to a (N, C) tensor.

  x: shape (N, C)
  y: returned tensor (same shape)
*/
static inline void layernorm_forward_2d(LayerNorm* ln, const Tensor* x, Tensor* y) {
    int N = x->shape[0];
    int C = x->shape[1];

    if (C != ln->normalized_dim) {
        printf("layernorm_forward_2d: ERROR: dim mismatch: got %d, expected %d\n",
               C, ln->normalized_dim);
        return;
    }

    int y_shape[2] = {N, C};
    tensor_init(y, 2, y_shape);

    for (int n = 0; n < N; ++n) {
        // 1. Compute mean
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) {
            mean += tensor_get2(x, n, c);
        }
        mean /= C;

        // 2. Compute variance
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float diff = tensor_get2(x, n, c) - mean;
            var += diff * diff;
        }
        var /= C;

        // 3. Normalize + scale + shift
        float inv_std = 1.0f / sqrtf(var + ln->eps);

        for (int c = 0; c < C; ++c) {
            float xc = tensor_get2(x, n, c);
            float g  = tensor_get1(&ln->gamma, c);
            float b  = tensor_get1(&ln->beta,  c);

            float norm = (xc - mean) * inv_std;
            tensor_set2(y, n, c, norm * g + b);
        }
    }
    
    // Create context for autograd
    LayerNormContext* ctx = (LayerNormContext*)malloc(sizeof(LayerNormContext));
    ctx->ln_layer = ln;
    ctx->input = (Tensor*)x;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = layernorm_backward_2d;
}


/*
  Apply LayerNorm to a (B, T, C) tensor.

  x: shape (B, T, C)
  y: returned tensor (same shape)
*/
static inline void layernorm_forward_3d(LayerNorm* ln, const Tensor* x, Tensor* y) {
    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    if (C != ln->normalized_dim) {
        printf("layernorm_forward_3d: ERROR: dim mismatch: got %d, expected %d\n",
               C, ln->normalized_dim);
        return;
    }

    int y_shape[3] = {B, T, C};
    tensor_init(y, 3, y_shape);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // 1. mean
            float mean = 0.0f;
            for (int c = 0; c < C; ++c)
                mean += tensor_get3(x, b, t, c);
            mean /= C;

            // 2. variance
            float var = 0.0f;
            for (int c = 0; c < C; ++c) {
                float diff = tensor_get3(x, b, t, c) - mean;
                var += diff * diff;
            }
            var /= C;

            float inv_std = 1.0f / sqrtf(var + ln->eps);

            // 3. normalize
            for (int c = 0; c < C; ++c) {
                float xc = tensor_get3(x, b, t, c);
                float g  = tensor_get1(&ln->gamma, c);
                float bb = tensor_get1(&ln->beta,  c);

                float norm = (xc - mean) * inv_std;
                tensor_set3(y, b, t, c, norm * g + bb);
            }
        }
    }
    
    // Create context for autograd
    LayerNormContext* ctx = (LayerNormContext*)malloc(sizeof(LayerNormContext));
    ctx->ln_layer = ln;
    ctx->input = (Tensor*)x;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = layernorm_backward_3d;
}


/*
  Dispatch based on ndim.
*/
static inline void layernorm_forward(LayerNorm* ln, const Tensor* x, Tensor* y) {
    if (x->ndim == 2) {
        layernorm_forward_2d(ln, x, y);
    } else if (x->ndim == 3) {
        layernorm_forward_3d(ln, x, y);
    } else {
        printf("layernorm_forward: ERROR: ndim must be 2 or 3\n");
    }
}
