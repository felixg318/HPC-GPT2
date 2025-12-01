// softmax.h
// Numerically stable softmax along the LAST dimension.

#pragma once

#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <mpi.h>

static int softmax_rank = 0;
static int softmax_world_size = 1;

static inline void softmax_set_distributed(int rank, int world_size) {
    softmax_rank = rank;
    softmax_world_size = (world_size > 0) ? world_size : 1;
}

static inline void softmax_column_range(int total_cols, int* start, int* end) {
    if (softmax_world_size <= 1 || total_cols <= 0) {
        *start = 0;
        *end = total_cols;
        return;
    }
    int base = total_cols / softmax_world_size;
    int extra = total_cols % softmax_world_size;
    int local = base + (softmax_rank < extra ? 1 : 0);
    int offset = softmax_rank * base + (softmax_rank < extra ? softmax_rank : extra);
    *start = offset;
    *end = offset + local;
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

    int col_start, col_end;
    softmax_column_range(C, &col_start, &col_end);

    for (int n = 0; n < N; ++n) {
        // Compute dot product of y and grad_y for this row
        float local_dot = 0.0f;
        for (int c = col_start; c < col_end; ++c) {
            local_dot += tensor_get2(y, n, c) * y->grad[tensor_index2(y, n, c)];
        }
        float dot_product = local_dot;
        if (softmax_world_size > 1) {
            MPI_Allreduce(&local_dot, &dot_product, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        
        // Compute grad_x for this row
        for (int c = col_start; c < col_end; ++c) {
            float y_val = tensor_get2(y, n, c);
            float grad_y_val = y->grad[tensor_index2(y, n, c)];
            x->grad[tensor_index2(x, n, c)] += y_val * (grad_y_val - dot_product);
        }
    }
    if (softmax_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, x->grad, tensor_numel(x), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
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

    int col_start, col_end;
    softmax_column_range(C, &col_start, &col_end);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // Compute dot product of y and grad_y for this slice
            float local_dot = 0.0f;
            for (int c = col_start; c < col_end; ++c) {
                local_dot += tensor_get3(y, b, t, c) * y->grad[tensor_index3(y, b, t, c)];
            }
            float dot_product = local_dot;
            if (softmax_world_size > 1) {
                MPI_Allreduce(&local_dot, &dot_product, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }
            
            // Compute grad_x for this slice
            for (int c = col_start; c < col_end; ++c) {
                float y_val = tensor_get3(y, b, t, c);
                float grad_y_val = y->grad[tensor_index3(y, b, t, c)];
                x->grad[tensor_index3(x, b, t, c)] += y_val * (grad_y_val - dot_product);
            }
        }
    }
    if (softmax_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, x->grad, tensor_numel(x), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
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
    tensor_zero(y);

    int col_start, col_end;
    softmax_column_range(C, &col_start, &col_end);

    for (int n = 0; n < N; ++n) {
        // 1. find max for stability
        float local_max = -1e30f;
        for (int c = col_start; c < col_end; ++c) {
            float v = tensor_get2(x, n, c);
            if (v > local_max) local_max = v;
        }
        float maxv = local_max;
        if (softmax_world_size > 1) {
            MPI_Allreduce(&local_max, &maxv, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        }

        // 2. compute exp(x - maxv) and sum
        float local_sum = 0.0f;
        for (int c = col_start; c < col_end; ++c) {
            float ex = expf(tensor_get2(x, n, c) - maxv);
            tensor_set2(y, n, c, ex);
            local_sum += ex;
        }
        float sum = local_sum;
        if (softmax_world_size > 1) {
            MPI_Allreduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        // 3. normalize
        float inv_sum = 1.0f / sum;
        for (int c = col_start; c < col_end; ++c) {
            tensor_set2(y, n, c, tensor_get2(y, n, c) * inv_sum);
        }
    }

    if (softmax_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, y->data, tensor_numel(y), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
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
    tensor_zero(y);

    int col_start, col_end;
    softmax_column_range(C, &col_start, &col_end);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {

            // 1. find max
            float local_max = -1e30f;
            for (int c = col_start; c < col_end; ++c) {
                float v = tensor_get3(x, b, t, c);
                if (v > local_max) local_max = v;
            }
            float maxv = local_max;
            if (softmax_world_size > 1) {
                MPI_Allreduce(&local_max, &maxv, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            }

            // 2. compute exps
            float local_sum = 0.0f;
            for (int c = col_start; c < col_end; ++c) {
                float ex = expf(tensor_get3(x, b, t, c) - maxv);
                tensor_set3(y, b, t, c, ex);
                local_sum += ex;
            }
            float sum = local_sum;
            if (softmax_world_size > 1) {
                MPI_Allreduce(&local_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            }

            float inv_sum = 1.0f / sum;

            // 3. normalize
            for (int c = col_start; c < col_end; ++c) {
                tensor_set3(y, b, t, c, tensor_get3(y, b, t, c) * inv_sum);
            }
        }
    }

    if (softmax_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, y->data, tensor_numel(y), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
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
