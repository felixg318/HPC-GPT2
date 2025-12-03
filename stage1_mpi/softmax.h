// softmax.h
// Numerically stable softmax along the LAST dimension.

#pragma once

#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

static int g_softmax_rank = 0;
static int g_softmax_world = 1;
static int g_softmax_last_dim_sharded = 0;

static inline void softmax_set_distributed(int rank, int world_size, int last_dim_sharded) {
    g_softmax_rank = (rank >= 0) ? rank : 0;
    g_softmax_world = (world_size > 0) ? world_size : 1;
    g_softmax_last_dim_sharded = last_dim_sharded ? 1 : 0;
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
    int rows = N;
    int distributed = g_softmax_last_dim_sharded && g_softmax_world > 1;

    float* local_dot = (float*)malloc(rows * sizeof(float));
    float* global_dot = NULL;
    if (local_dot == NULL) {
        printf("softmax_backward_2d: failed to allocate buffer\n");
        free(ctx);
        return;
    }
    if (distributed) {
        global_dot = (float*)malloc(rows * sizeof(float));
        if (global_dot == NULL) {
            printf("softmax_backward_2d: failed to allocate global buffer\n");
            free(local_dot);
            free(ctx);
            return;
        }
    } else {
        global_dot = local_dot;
    }

    for (int n = 0; n < N; ++n) {
        float dot_product = 0.0f;
        for (int c = 0; c < C; ++c) {
            dot_product += tensor_get2(y, n, c) * y->grad[tensor_index2(y, n, c)];
        }
        local_dot[n] = dot_product;
    }

    if (distributed) {
        MPI_Allreduce(local_dot, global_dot, rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    for (int n = 0; n < N; ++n) {
        float dot_product = global_dot[n];
        for (int c = 0; c < C; ++c) {
            float y_val = tensor_get2(y, n, c);
            float grad_y_val = y->grad[tensor_index2(y, n, c)];
            x->grad[tensor_index2(x, n, c)] += y_val * (grad_y_val - dot_product);
        }
    }

    if (global_dot != local_dot) free(global_dot);
    free(local_dot);
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
    int rows = B * T;
    int distributed = g_softmax_last_dim_sharded && g_softmax_world > 1;

    float* local_dot = (float*)malloc(rows * sizeof(float));
    float* global_dot = NULL;
    if (local_dot == NULL) {
        printf("softmax_backward_3d: failed to allocate buffer\n");
        free(ctx);
        return;
    }
    if (distributed) {
        global_dot = (float*)malloc(rows * sizeof(float));
        if (global_dot == NULL) {
            printf("softmax_backward_3d: failed to allocate global buffer\n");
            free(local_dot);
            free(ctx);
            return;
        }
    } else {
        global_dot = local_dot;
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float dot_product = 0.0f;
            for (int c = 0; c < C; ++c) {
                dot_product += tensor_get3(y, b, t, c) * y->grad[tensor_index3(y, b, t, c)];
            }
            local_dot[b * T + t] = dot_product;
        }
    }

    if (distributed) {
        MPI_Allreduce(local_dot, global_dot, rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            float dot_product = global_dot[b * T + t];
            for (int c = 0; c < C; ++c) {
                float y_val = tensor_get3(y, b, t, c);
                float grad_y_val = y->grad[tensor_index3(y, b, t, c)];
                x->grad[tensor_index3(x, b, t, c)] += y_val * (grad_y_val - dot_product);
            }
        }
    }

    if (global_dot != local_dot) free(global_dot);
    free(local_dot);
    free(ctx);
}


/*
    Softmax for a (N, C) tensor over last dimension C.
*/
static inline void softmax_forward_2d(const Tensor* x, Tensor* y) {
    int N = x->shape[0];
    int C = x->shape[1];
    int rows = N;
    int distributed = g_softmax_last_dim_sharded && g_softmax_world > 1;

    int y_shape[2] = {N, C};
    tensor_init(y, 2, y_shape);

    float* local_max = (float*)malloc(rows * sizeof(float));
    float* local_sum = (float*)malloc(rows * sizeof(float));
    float* global_max = NULL;
    float* global_sum = NULL;
    if (local_max == NULL || local_sum == NULL) {
        printf("softmax_forward_2d: failed to allocate buffers\n");
        free(local_max);
        free(local_sum);
        return;
    }
    if (distributed) {
        global_max = (float*)malloc(rows * sizeof(float));
        global_sum = (float*)malloc(rows * sizeof(float));
        if (global_max == NULL || global_sum == NULL) {
            printf("softmax_forward_2d: failed to allocate global buffers\n");
            free(local_max);
            free(local_sum);
            free(global_max);
            free(global_sum);
            return;
        }
    } else {
        global_max = local_max;
        global_sum = local_sum;
    }

    for (int n = 0; n < N; ++n) {
        float maxv = -1e30f;
        for (int c = 0; c < C; ++c) {
            float v = tensor_get2(x, n, c);
            if (v > maxv) maxv = v;
        }
        local_max[n] = maxv;
    }

    if (distributed) {
        MPI_Allreduce(local_max, global_max, rows, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }

    for (int n = 0; n < N; ++n) {
        float base = global_max[n];
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += expf(tensor_get2(x, n, c) - base);
        }
        local_sum[n] = sum;
    }

    if (distributed) {
        MPI_Allreduce(local_sum, global_sum, rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    for (int n = 0; n < N; ++n) {
        float base = global_max[n];
        float denom = global_sum[n];
        float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        for (int c = 0; c < C; ++c) {
            float ex = expf(tensor_get2(x, n, c) - base);
            tensor_set2(y, n, c, ex * inv_sum);
        }
    }

    if (global_max != local_max) free(global_max);
    if (global_sum != local_sum) free(global_sum);
    free(local_max);
    free(local_sum);
    
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
    int rows = B * T;
    int distributed = g_softmax_last_dim_sharded && g_softmax_world > 1;

    int y_shape[3] = {B, T, C};
    tensor_init(y, 3, y_shape);

    float* local_max = (float*)malloc(rows * sizeof(float));
    float* local_sum = (float*)malloc(rows * sizeof(float));
    float* global_max = NULL;
    float* global_sum = NULL;
    if (local_max == NULL || local_sum == NULL) {
        printf("softmax_forward_3d: failed to allocate buffers\n");
        free(local_max);
        free(local_sum);
        return;
    }
    if (distributed) {
        global_max = (float*)malloc(rows * sizeof(float));
        global_sum = (float*)malloc(rows * sizeof(float));
        if (global_max == NULL || global_sum == NULL) {
            printf("softmax_forward_3d: failed to allocate global buffers\n");
            free(local_max);
            free(local_sum);
            free(global_max);
            free(global_sum);
            return;
        }
    } else {
        global_max = local_max;
        global_sum = local_sum;
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int row = b * T + t;
            float maxv = -1e30f;
            for (int c = 0; c < C; ++c) {
                float v = tensor_get3(x, b, t, c);
                if (v > maxv) maxv = v;
            }
            local_max[row] = maxv;
        }
    }

    if (distributed) {
        MPI_Allreduce(local_max, global_max, rows, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int row = b * T + t;
            float base = global_max[row];
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += expf(tensor_get3(x, b, t, c) - base);
            }
            local_sum[row] = sum;
        }
    }

    if (distributed) {
        MPI_Allreduce(local_sum, global_sum, rows, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int row = b * T + t;
            float base = global_max[row];
            float denom = global_sum[row];
            float inv_sum = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
            for (int c = 0; c < C; ++c) {
                float ex = expf(tensor_get3(x, b, t, c) - base);
                tensor_set3(y, b, t, c, ex * inv_sum);
            }
        }
    }

    if (global_max != local_max) free(global_max);
    if (global_sum != local_sum) free(global_sum);
    free(local_max);
    free(local_sum);
    
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
