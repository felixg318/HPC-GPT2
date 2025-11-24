// add.h
// Add operation.

#pragma once

#include "tensor.h"

// Context for the add operation
typedef struct {
    Tensor* a;
    Tensor* b;
} AddContext;


// Backward function for the add operation
static inline void add_backward(Tensor* t) {
    AddContext* ctx = (AddContext*)t->_ctx;
    Tensor* a = ctx->a;
    Tensor* b = ctx->b;
    
    int n = tensor_numel(t);
    for (int i = 0; i < n; ++i) {
        a->grad[i] += t->grad[i];
        b->grad[i] += t->grad[i];
    }
    free(ctx);
}


/*
  Elementwise add of two tensors with same shape.

  out = a + b

  Assumes:
    - same ndim
    - same shape
*/
static inline void add_forward(const Tensor* a, const Tensor* b, Tensor* out) {
    if (a->ndim != b->ndim) {
        printf("add_forward: ERROR: ndim mismatch\n");
        return;
    }

    // Copy shape
    int shape[TENSOR_MAX_DIMS];
    for (int i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) {
            printf("add_forward: ERROR: shape[%d] mismatch: %d vs %d\n",
                   i, a->shape[i], b->shape[i]);
            return;
        }
        shape[i] = a->shape[i];
    }

    // Init out with same shape
    tensor_init(out, a->ndim, shape);

    int n = tensor_numel(a);
    for (int i = 0; i < n; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    // Create context for autograd
    AddContext* ctx = (AddContext*)malloc(sizeof(AddContext));
    ctx->a = (Tensor*)a;
    ctx->b = (Tensor*)b;
    
    out->_ctx = ctx;
    out->_backward = add_backward;
}
