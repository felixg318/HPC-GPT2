// matmul.h
// Matrix multiplication operation.

#pragma once

#include "tensor.h"

// Context for the matmul operation
typedef struct {
    Tensor* a;
    Tensor* b;
} MatmulContext;

// Backward function for matrix multiplication
static inline void matmul_backward(Tensor* t) {
    MatmulContext* ctx = (MatmulContext*)t->_ctx;
    Tensor* a = ctx->a;
    Tensor* b = ctx->b;
    Tensor* c = t; // output tensor

    // grad_a = grad_c @ b^T
    // grad_b = a^T @ grad_c

    // This is a simplified implementation for specific shapes used in attention.
    // A general matmul backward is more complex.

    // Let's assume a is (B, T, C1) and b is (B, C1, C2) -> c is (B, T, C2)
    if (a->ndim == 3 && b->ndim == 3 && c->ndim == 3 && a->shape[0] == b->shape[0]) {
        int B = a->shape[0];
        int T = a->shape[1];
        int C1 = a->shape[2];
        int C2 = b->shape[2];

        // grad_a
        for (int b_ = 0; b_ < B; ++b_) {
            for (int t_ = 0; t_ < T; ++t_) {
                for (int c1_ = 0; c1_ < C1; ++c1_) {
                    float grad_sum = 0.0f;
                    for (int c2_ = 0; c2_ < C2; ++c2_) {
                        grad_sum += c->grad[tensor_index3(c, b_, t_, c2_)] * tensor_get3(b, b_, c1_, c2_);
                    }
                    a->grad[tensor_index3(a, b_, t_, c1_)] += grad_sum;
                }
            }
        }

        // grad_b
        for (int b_ = 0; b_ < B; ++b_) {
            for (int c1_ = 0; c1_ < C1; ++c1_) {
                for (int c2_ = 0; c2_ < C2; ++c2_) {
                    float grad_sum = 0.0f;
                    for (int t_ = 0; t_ < T; ++t_) {
                        grad_sum += a->grad[tensor_index3(a, b_, t_, c1_)] * c->grad[tensor_index3(c, b_, t_, c2_)];
                    }
                    b->grad[tensor_index3(b, b_, c1_, c2_)] += grad_sum;
                }
            }
        }
    } else {
        printf("matmul_backward: ERROR: unsupported shapes\n");
    }

    free(ctx);
}


/*
  Matrix multiplication of two tensors.
  c = a @ b
*/
static inline void matmul_forward(const Tensor* a, const Tensor* b, Tensor* out) {
    if (a->ndim == 3 && b->ndim == 3 && a->shape[0] == b->shape[0] && a->shape[2] == b->shape[1]) {
        // Batch matmul for 3D tensors (B, T, C1) @ (B, C1, C2) -> (B, T, C2)
        // This is a specific case for attention
        int B = a->shape[0];
        int T = a->shape[1];
        int C1 = a->shape[2];
        int C2 = b->shape[2];

        int out_shape[3] = {B, T, C2};
        tensor_init(out, 3, out_shape);

        for (int b_ = 0; b_ < B; ++b_) {
            for (int t_ = 0; t_ < T; ++t_) {
                for (int c2_ = 0; c2_ < C2; ++c2_) {
                    float sum = 0.0f;
                    for (int c1_ = 0; c1_ < C1; ++c1_) {
                        sum += tensor_get3(a, b_, t_, c1_) * tensor_get3(b, b_, c1_, c2_);
                    }
                    tensor_set3(out, b_, t_, c2_, sum);
                }
            }
        }
    } else if (a->ndim == 2 && b->ndim == 2 && a->shape[1] == b->shape[0]) {
        // Matmul for 2D tensors (N, C1) @ (C1, C2) -> (N, C2)
        int N = a->shape[0];
        int C1 = a->shape[1];
        int C2 = b->shape[1];

        int out_shape[2] = {N, C2};
        tensor_init(out, 2, out_shape);

        for (int n = 0; n < N; ++n) {
            for (int c2 = 0; c2 < C2; ++c2) {
                float sum = 0.0f;
                for (int c1 = 0; c1 < C1; ++c1) {
                    sum += tensor_get2(a, n, c1) * tensor_get2(b, c1, c2);
                }
                tensor_set2(out, n, c2, sum);
            }
        }
    }
    else {
        printf("matmul_forward: ERROR: unsupported shapes\n");
        return;
    }

    // Create context for autograd
    MatmulContext* ctx = (MatmulContext*)malloc(sizeof(MatmulContext));
    ctx->a = (Tensor*)a;
    ctx->b = (Tensor*)b;

    tensor_set_inputs2(out, (Tensor*)a, (Tensor*)b);
    out->_ctx = ctx;
    out->_backward = matmul_backward;
}
