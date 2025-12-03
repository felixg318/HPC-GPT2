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

    // The backward of a transpose is transposing the gradient
    if (y->ndim != 3 || x->ndim != 3) {
        printf("transpose_backward: ERROR: only 3D tensors are supported\n");
        free(ctx);
        return;
    }

    int B = x->shape[0];
    int T1 = x->shape[1];
    int T2 = x->shape[2];

    int used_cuda = 0;
#ifdef USE_CUDA
    used_cuda = cuda_transpose_last2_backward(y->grad, x->grad, B, T1, T2);
#endif

    if (!used_cuda) {
        for (int b = 0; b < B; ++b) {
            for (int t1 = 0; t1 < T1; ++t1) {
                for (int t2 = 0; t2 < T2; ++t2) {
                    int y_idx = tensor_index3(y, b, t2, t1);
                    int x_idx = tensor_index3(x, b, t1, t2);
                    x->grad[x_idx] += y->grad[y_idx];
                }
            }
        }
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

    int used_cuda = 0;
#ifdef USE_CUDA
    used_cuda = cuda_transpose_last2(x->data, y->data, B, T1, T2);
#endif

    if (!used_cuda) {
        for (int b = 0; b < B; ++b) {
            for (int t1 = 0; t1 < T1; ++t1) {
                for (int t2 = 0; t2 < T2; ++t2) {
                    tensor_set3(y, b, t2, t1, tensor_get3(x, b, t1, t2));
                }
            }
        }
    }
    
    // Create context for autograd
    TransposeContext* ctx = (TransposeContext*)malloc(sizeof(TransposeContext));
    ctx->input = (Tensor*)x;
    
    y->_ctx = ctx;
    y->_backward = transpose_backward;
}
