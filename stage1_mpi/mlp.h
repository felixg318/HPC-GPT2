// mlp.h
#pragma once

#include "tensor.h"
#include "linear.h"
#include "gelu.h"
#include <stdio.h>
#include <stdlib.h>

/*
   MLP struct (mirrors PyTorch MLP):

   hidden_dim = 4 * n_embd

   forward:
       x -> c_fc (Linear) -> GELU -> c_proj (Linear) -> dropout(optional)
*/
typedef struct {
    Linear c_fc;      // (n_embd -> hidden_dim)
    Linear c_proj;    // (hidden_dim -> n_embd)
    float dropout_p;  // keep probability (unused for now)
} MLP;



/*
  Initialize the MLP.

  Arguments:
    mlp        : pointer to MLP struct
    n_embd     : embedding dimension (e.g. 768 for GPT-2 small)
    dropout_p  : dropout probability (we will ignore for now)
*/
static inline void mlp_init(MLP* mlp, int n_embd, float dropout_p) {
    int hidden_dim = 4 * n_embd;

    // Initialize c_fc: (n_embd -> hidden_dim)
    linear_init(&mlp->c_fc, n_embd, hidden_dim, 1 /* use_bias */);

    // Initialize c_proj: (hidden_dim -> n_embd)
    linear_init(&mlp->c_proj, hidden_dim, n_embd, 1 /* use_bias */);

    mlp->dropout_p = dropout_p;
}


/*
  Free MLP internal memory.
*/
static inline void mlp_free(MLP* mlp) {
    linear_free(&mlp->c_fc);
    linear_free(&mlp->c_proj);
}

static inline void mlp_collect_params(MLP* mlp, TensorPtrArray* list) {
    linear_collect_params(&mlp->c_fc, list);
    linear_collect_params(&mlp->c_proj, list);
}

/*
  Forward pass through the MLP.

  Input:
    x  : Tensor with shape (B, T, n_embd) or (N, n_embd)

  Output:
    y  : Newly allocated tensor with same shape as x (B,T,n_embd)

  NOTE:
    x is NOT modified and caller must free y with tensor_free().
*/
static inline void mlp_forward(MLP* mlp,
                               const Tensor* x,
                               Tensor* y,
                               TensorTracker* tracker) {
    Tensor* t1 = tensor_tracker_new(tracker);   // result of first linear
    Tensor* t2 = tensor_tracker_new(tracker);   // result of GELU(t1)

    // 1) x -> c_fc
    linear_forward(&mlp->c_fc, x, t1);

    // 2) GELU
    gelu_tensor(t1, t2);

    // 3) t2 -> c_proj
    linear_forward(&mlp->c_proj, t2, y);
}
