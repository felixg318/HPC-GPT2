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

// Context for the MLP operation
typedef struct {
    MLP* mlp;
    TensorTracker* tracker;
    Tensor* x;
    Tensor* t1;
    Tensor* t2;
} MLPContext;


/*
  Simple helper to fill Linear weights & bias with constants.
  This is just for testing; later - load real weights or random init.
*/
static inline void linear_fill_constant(Linear* layer, float w_val, float b_val) {
    tensor_fill(&layer->weight, w_val);
    if (layer->use_bias) {
        tensor_fill(&layer->bias, b_val);
    }
}

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

    // ---- Debug initialization so it's not all zeros ----
    // For now: c_fc weights = 0.1, bias = 0.01
    //          c_proj weights = 0.05, bias = 0.0
    linear_fill_constant(&mlp->c_fc, 0.1f, 0.01f);
    linear_fill_constant(&mlp->c_proj, 0.05f, 0.0f);
}


/*
  Free MLP internal memory.
*/
static inline void mlp_free(MLP* mlp) {
    linear_free(&mlp->c_fc);
    linear_free(&mlp->c_proj);
}

// Backward function for MLP
static inline void mlp_backward(Tensor* t) {
    MLPContext* ctx = (MLPContext*)t->_ctx;

    // Backward pass for c_proj
    ctx->t2->_backward(t);

    // Backward pass for GELU
    ctx->t1->_backward(ctx->t2);

    // Backward pass for c_fc
    ctx->x->_backward(ctx->t1);

    if (ctx->tracker == NULL) {
        tensor_free(ctx->t1);
        free(ctx->t1);
        tensor_free(ctx->t2);
        free(ctx->t2);
    }
    free(ctx);
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

    // Create context for autograd
    MLPContext* ctx = (MLPContext*)malloc(sizeof(MLPContext));
    ctx->mlp = mlp;
    ctx->tracker = tracker;
    ctx->x = (Tensor*)x;
    ctx->t1 = t1;
    ctx->t2 = t2;
    
    y->_ctx = ctx;
    y->_backward = mlp_backward;
}
