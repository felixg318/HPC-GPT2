// gelu.h
// GELU activation (approximate, tanh-based), applied to scalar or whole Tensor.

#pragma once

#include <math.h>   // for tanh, sqrt, powf
#include <mpi.h>
#include "tensor.h"

static int gelu_rank = 0;
static int gelu_world_size = 1;

static inline void gelu_set_distributed(int rank, int world_size) {
    gelu_rank = rank;
    gelu_world_size = (world_size > 0) ? world_size : 1;
}

static inline void gelu_index_range(int total, int* start, int* end) {
    if (gelu_world_size <= 1 || total <= 0) {
        *start = 0;
        *end = total;
        return;
    }
    int base = total / gelu_world_size;
    int extra = total % gelu_world_size;
    int local = base + (gelu_rank < extra ? 1 : 0);
    int offset = gelu_rank * base + (gelu_rank < extra ? gelu_rank : extra);
    *start = offset;
    *end = offset + local;
}

// Context for the GELU operation
typedef struct {
    Tensor* input;
} GELUContext;

/*
  Scalar GELU (approximate version, like PyTorch GELU with approximate='tanh').

  gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
*/
static inline float gelu_scalar(float x) {
    // sqrt(2/pi) is ~0.79788456, we can precompute it
    const float SQRT_2_OVER_PI = 0.7978845608028654f;

    float x3 = x * x * x;  // x^3
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
    float t = tanhf(inner);  // hyperbolic tangent
    float y = 0.5f * x * (1.0f + t);
    return y;
}

// Backward function for GELU
static inline void gelu_backward(Tensor* t) {
    GELUContext* ctx = (GELUContext*)t->_ctx;
    Tensor* x = ctx->input;
    Tensor* y = t;

    const float SQRT_2_OVER_PI = 0.7978845608028654f;

    int n = tensor_numel(x);
    int start, end;
    gelu_index_range(n, &start, &end);
    for (int i = start; i < end; ++i) {
        float x_val = x->data[i];
        float x2 = x_val * x_val;
        float x3 = x2 * x_val;
        
        float inner = SQRT_2_OVER_PI * (x_val + 0.044715f * x3);
        float t = tanhf(inner);
        float sech_inner_2 = 1.0f - t * t;
        float inner_derivative = SQRT_2_OVER_PI * (1.0f + 0.044715f * 3.0f * x2);
        
        float grad_gelu = 0.5f * (1.0f + t) + 0.5f * x_val * sech_inner_2 * inner_derivative;
        x->grad[i] += y->grad[i] * grad_gelu;
    }
    if (gelu_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, x->grad, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    free(ctx);
}


/*
  Apply GELU in-place to all elements of a Tensor.

  That is: t->data[i] = gelu_scalar(t->data[i]) for all i.
*/
static inline void gelu_inplace(Tensor* t) {
    int n = tensor_numel(t);
    int start, end;
    gelu_index_range(n, &start, &end);
    for (int i = start; i < end; ++i) {
        float x = t->data[i];
        t->data[i] = gelu_scalar(x);
    }
    if (gelu_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, t->data, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
}

/*
  Apply GELU to input tensor x, write result into output tensor y.

  - x: input tensor (any shape)
  - y: output tensor (allocated here, same shape as x)

  This is NOT in-place: x is unchanged, y is new.
*/
static inline void gelu_tensor(const Tensor* x, Tensor* y) {
    // Copy shape from x to y
    int shape[ TENSOR_MAX_DIMS ];
    for (int i = 0; i < x->ndim; ++i) {
        shape[i] = x->shape[i];
    }
    // Initialize y with same ndim and shape
    tensor_init(y, x->ndim, shape);

    int n = tensor_numel(x);
    tensor_zero(y);
    int start, end;
    gelu_index_range(n, &start, &end);
    for (int i = start; i < end; ++i) {
        float v = x->data[i];
        y->data[i] = gelu_scalar(v);
    }
    if (gelu_world_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, y->data, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    // Create context for autograd
    GELUContext* ctx = (GELUContext*)malloc(sizeof(GELUContext));
    ctx->input = (Tensor*)x;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = gelu_backward;
}
