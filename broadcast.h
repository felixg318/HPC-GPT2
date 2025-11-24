// broadcast.h
// Broadcasted add operation.

#pragma once

#include "tensor.h"
#include "add.h"

// Context for the broadcast add operation
typedef struct {
    Tensor* a;
    Tensor* b;
} BroadcastAddContext;

// Backward function for broadcast add
static inline void broadcast_add_backward(Tensor* t) {
    BroadcastAddContext* ctx = (BroadcastAddContext*)t->_ctx;
    Tensor* a = ctx->a;
    Tensor* b = ctx->b;
    Tensor* out = t;

    // The gradient of a broadcasted add is to sum the gradient of the output
    // along the broadcasted dimensions.
    
    // a is (B, T, C), b is (T, C) -> out is (B, T, C)
    if (a->ndim == 3 && b->ndim == 2) {
        int B = a->shape[0];
        int T = a->shape[1];
        int C = a->shape[2];
        
        // grad_a is just grad_out
        for(int i=0; i<tensor_numel(a); ++i) a->grad[i] += out->grad[i];
        
        // grad_b is sum(grad_out, axis=0)
        for(int t_ = 0; t_ < T; ++t_) {
            for(int c_ = 0; c_ < C; ++c_) {
                float sum = 0.0f;
                for(int b_ = 0; b_ < B; ++b_) {
                    sum += out->grad[tensor_index3(out, b_, t_, c_)];
                }
                b->grad[tensor_index2(b, t_, c_)] += sum;
            }
        }
    } else {
        printf("broadcast_add_backward: ERROR: unsupported shapes\n");
    }

    free(ctx);
}

/*
  Broadcasted add of two tensors.
  a: (B, T, C)
  b: (T, C)
  out: (B, T, C)
*/
static inline void broadcast_add_forward(const Tensor* a, const Tensor* b, Tensor* out) {
    if (a->ndim != 3 || b->ndim != 2) {
        printf("broadcast_add_forward: ERROR: unsupported shapes\n");
        return;
    }
    if (a->shape[1] != b->shape[0] || a->shape[2] != b->shape[1]) {
        printf("broadcast_add_forward: ERROR: shape mismatch\n");
        return;
    }

    tensor_init(out, a->ndim, a->shape);

    int B = a->shape[0];
    int T = a->shape[1];
    int C = a->shape[2];

    for (int b_ = 0; b_ < B; ++b_) {
        for (int t_ = 0; t_ < T; ++t_) {
            for (int c_ = 0; c_ < C; ++c_) {
                tensor_set3(out, b_, t_, c_, tensor_get3(a, b_, t_, c_) + tensor_get2(b, t_, c_));
            }
        }
    }

    // Create context for autograd
    BroadcastAddContext* ctx = (BroadcastAddContext*)malloc(sizeof(BroadcastAddContext));
    ctx->a = (Tensor*)a;
    ctx->b = (Tensor*)b;

    out->_ctx = ctx;
    out->_backward = broadcast_add_backward;
}
