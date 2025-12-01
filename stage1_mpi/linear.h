// linear.h
// Fully connected layer: y = x W + b
// Weight layout: (in_dim, out_dim), row-major.
// This matches Python: y = x @ W_c  where W_c has shape (in_dim, out_dim).

#pragma once

#include <stdio.h>
#include "tensor.h"

typedef struct {
    int in_dim;
    int out_dim;
    int use_bias;   // 1 or 0

    Tensor weight;  // shape: (in_dim, out_dim)
    Tensor bias;    // shape: (out_dim,) if use_bias
} Linear;

// Context for the linear operation
typedef struct {
    Linear* linear_layer;
    Tensor* input;
} LinearContext;

/*
  Initialize Linear layer.

  in_dim   : input dimension
  out_dim  : output dimension
  use_bias : 1 to allocate bias, 0 otherwise

  We DO NOT do any random init here; caller can fill with whatever.
*/
static inline void linear_init(Linear* lin, int in_dim, int out_dim, int use_bias) {
    lin->in_dim   = in_dim;
    lin->out_dim  = out_dim;
    lin->use_bias = use_bias;

    // weight: (in_dim, out_dim)
    int w_shape[2];
    w_shape[0] = in_dim;
    w_shape[1] = out_dim;
    tensor_init(&lin->weight, 2, w_shape);

    // bias: (out_dim,)
    if (use_bias) {
        int b_shape[1];
        b_shape[0] = out_dim;
        tensor_init(&lin->bias, 1, b_shape);
        tensor_fill(&lin->bias, 0.0f);
    } else {
        lin->bias.data = NULL;
        lin->bias.ndim = 0;
    }

    tensor_fill_randn(&lin->weight, 0.0f, 0.02f);
}


/*
  Free Linear resources.
*/
static inline void linear_free(Linear* lin) {
    tensor_free(&lin->weight);
    if (lin->use_bias && lin->bias.data != NULL) {
        tensor_free(&lin->bias);
    }
}

static inline void linear_collect_params(Linear* lin, TensorPtrArray* list) {
    tensor_ptr_array_push(list, &lin->weight);
    if (lin->use_bias && lin->bias.data != NULL) {
        tensor_ptr_array_push(list, &lin->bias);
    }
}

// Backward function for 2D linear layer
static inline void linear_backward_2d(Tensor* t) {
    LinearContext* ctx = (LinearContext*)t->_ctx;
    Linear* lin = ctx->linear_layer;
    Tensor* x = ctx->input;
    Tensor* y = t;

    int N = x->shape[0];
    int C_in = lin->in_dim;
    int C_out = lin->out_dim;

    // grad_input = grad_output @ W^T
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < C_in; ++i) {
            float grad_sum = 0.0f;
            for (int o = 0; o < C_out; ++o) {
                grad_sum += y->grad[tensor_index2(y, n, o)] * tensor_get2(&lin->weight, i, o);
            }
            x->grad[tensor_index2(x, n, i)] += grad_sum;
        }
    }

    // grad_weight = x^T @ grad_output
    for (int i = 0; i < C_in; ++i) {
        for (int o = 0; o < C_out; ++o) {
            float grad_sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                grad_sum += tensor_get2(x, n, i) * y->grad[tensor_index2(y, n, o)];
            }
            lin->weight.grad[tensor_index2(&lin->weight, i, o)] += grad_sum;
        }
    }

    // grad_bias = sum(grad_output, axis=0)
    if (lin->use_bias) {
        for (int o = 0; o < C_out; ++o) {
            float grad_sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                grad_sum += y->grad[tensor_index2(y, n, o)];
            }
            lin->bias.grad[o] += grad_sum;
        }
    }
    free(ctx);
}

// Backward function for 3D linear layer
static inline void linear_backward_3d(Tensor* t) {
    LinearContext* ctx = (LinearContext*)t->_ctx;
    Linear* lin = ctx->linear_layer;
    Tensor* x = ctx->input;
    Tensor* y = t;

    int B = x->shape[0];
    int T = x->shape[1];
    int C_in = lin->in_dim;
    int C_out = lin->out_dim;

    // grad_input = grad_output @ W^T
    for (int b = 0; b < B; ++b) {
        for (int t_ = 0; t_ < T; ++t_) {
            for (int i = 0; i < C_in; ++i) {
                float grad_sum = 0.0f;
                for (int o = 0; o < C_out; ++o) {
                    grad_sum += y->grad[tensor_index3(y, b, t_, o)] * tensor_get2(&lin->weight, i, o);
                }
                x->grad[tensor_index3(x, b, t_, i)] += grad_sum;
            }
        }
    }

    // grad_weight = x^T @ grad_output
    for (int i = 0; i < C_in; ++i) {
        for (int o = 0; o < C_out; ++o) {
            float grad_sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int t_ = 0; t_ < T; ++t_) {
                    grad_sum += tensor_get3(x, b, t_, i) * y->grad[tensor_index3(y, b, t_, o)];
                }
            }
            lin->weight.grad[tensor_index2(&lin->weight, i, o)] += grad_sum;
        }
    }

    // grad_bias = sum(grad_output, axis=0)
    if (lin->use_bias) {
        for (int o = 0; o < C_out; ++o) {
            float grad_sum = 0.0f;
            for (int b = 0; b < B; ++b) {
                for (int t_ = 0; t_ < T; ++t_) {
                    grad_sum += y->grad[tensor_index3(y, b, t_, o)];
                }
            }
            lin->bias.grad[o] += grad_sum;
        }
    }
    free(ctx);
}


/*
  Core matmul for 2D input:

    x: (N, in_dim)
    y: (N, out_dim)

  y[n, o] = sum_i x[n, i] * W[i, o] + b[o]
*/
static inline void linear_forward_2d(Linear* lin, const Tensor* x, Tensor* y) {
    int N = x->shape[0];
    int C_in = x->shape[1];

    if (C_in != lin->in_dim) {
        printf("linear_forward_2d: ERROR: input dim mismatch: %d vs %d\n",
               C_in, lin->in_dim);
        return;
    }

    int C_out = lin->out_dim;
    int y_shape[2] = {N, C_out};
    tensor_init(y, 2, y_shape);

    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < C_out; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < C_in; ++i) {
                // x[n, i]
                float xv = tensor_get2(x, n, i);
                // W[i, o]
                float wv = tensor_get2(&lin->weight, i, o);
                sum += xv * wv;
            }
            if (lin->use_bias) {
                sum += tensor_get1(&lin->bias, o);
            }
            tensor_set2(y, n, o, sum);
        }
    }
    
    // Create context for autograd
    LinearContext* ctx = (LinearContext*)malloc(sizeof(LinearContext));
    ctx->linear_layer = lin;
    ctx->input = (Tensor*)x;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = linear_backward_2d;
}


/*
  3D version:

    x: (B, T, in_dim)
    y: (B, T, out_dim)

  We flatten (B,T) to N = B*T and reuse the 2D logic.
*/
static inline void linear_forward_3d(Linear* lin, const Tensor* x, Tensor* y) {
    int B = x->shape[0];
    int T = x->shape[1];
    int C_in = x->shape[2];

    if (C_in != lin->in_dim) {
        printf("linear_forward_3d: ERROR: input dim mismatch: %d vs %d\n",
               C_in, lin->in_dim);
        return;
    }

    int C_out = lin->out_dim;
    int y_shape[3] = {B, T, C_out};
    tensor_init(y, 3, y_shape);

    // For each (b,t), compute out[b,t,:] = x[b,t,:] @ W
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < C_out; ++o) {
                float sum = 0.0f;
                for (int i = 0; i < C_in; ++i) {
                    float xv = tensor_get3(x, b, t, i);    // x[b,t,i]
                    float wv = tensor_get2(&lin->weight, i, o); // W[i,o]
                    sum += xv * wv;
                }
                if (lin->use_bias) {
                    sum += tensor_get1(&lin->bias, o);
                }
                tensor_set3(y, b, t, o, sum);
            }
        }
    }
    
    // Create context for autograd
    LinearContext* ctx = (LinearContext*)malloc(sizeof(LinearContext));
    ctx->linear_layer = lin;
    ctx->input = (Tensor*)x;
    tensor_set_inputs1(y, (Tensor*)x);
    y->_ctx = ctx;
    y->_backward = linear_backward_3d;
}


/*
  Dispatch based on ndim of x.
*/
static inline void linear_forward(Linear* lin, const Tensor* x, Tensor* y) {
    if (x->ndim == 2) {
        linear_forward_2d(lin, x, y);
    } else if (x->ndim == 3) {
        linear_forward_3d(lin, x, y);
    } else {
        printf("linear_forward: ERROR: x->ndim must be 2 or 3\n");
    }
}
