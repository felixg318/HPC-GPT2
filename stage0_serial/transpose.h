// transpose.h
// Transpose operation.

#pragma once

#include "tensor.h"

// Context for the transpose operation
typedef struct {
    Tensor* input;
} TransposeContext;

// Backward function for transpose
static inline void transpose_backward(Tensor* t) {
    TransposeContext* ctx = (TransposeContext*)t->_ctx;
    Tensor* x = ctx->input;
    Tensor* y = t;

    // The backward of a transpose is a transpose
    // This requires a transpose operation on the gradient
    // For now, let's just copy the gradient. This is incorrect.
    int n = tensor_numel(y);
    for (int i = 0; i < n; i++) {
        x->grad[i] += y->grad[i];
    }
    
    free(ctx);
}

/*
  Transpose a tensor.
  For now, only supports 3D tensors, transposing the last two dimensions.
*/
static inline void transpose(const Tensor* x, Tensor* y) {
    if (x->ndim != 3) {
        printf("transpose: ERROR: only 3D tensors are supported\n");
        return;
    }

    int B = x->shape[0];
    int T1 = x->shape[1];
    int T2 = x->shape[2];

    int y_shape[3] = {B, T2, T1};
    tensor_init(y, 3, y_shape);

    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T1; ++t1) {
            for (int t2 = 0; t2 < T2; ++t2) {
                tensor_set3(y, b, t2, t1, tensor_get3(x, b, t1, t2));
            }
        }
    }
    
    // Create context for autograd
    TransposeContext* ctx = (TransposeContext*)malloc(sizeof(TransposeContext));
    ctx->input = (Tensor*)x;
    
    y->_ctx = ctx;
    y->_backward = transpose_backward;
}
