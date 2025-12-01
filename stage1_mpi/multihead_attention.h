// multihead_attention.h
// Multi-head self-attention built from several Head structs + final Linear proj.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "linear.h"
#include "head.h"

typedef struct {
    int n_heads;       // number of heads
    int embed_dim;     // model dimension (n_embd)
    int head_size;     // embed_dim / n_heads
    float dropout_p;   // not used yet (inference only)

    Head* heads;       // array of Head of length n_heads
    Linear proj;       // final projection: (embed_dim -> embed_dim)
} MultiHeadAttention;

// Context for the multi-head attention operation
typedef struct {
    TensorTracker* tracker;
    int n_heads;
    int head_size;
    Tensor** head_outputs;
} ConcatHeadsContext;

static inline void concat_heads_backward(Tensor* t) {
    ConcatHeadsContext* ctx = (ConcatHeadsContext*)t->_ctx;
    Tensor* concat = t;
    if (ctx->n_heads == 0) {
        free(ctx->head_outputs);
        free(ctx);
        return;
    }
    Tensor* ref = ctx->head_outputs[0];
    int B = ref->shape[0];
    int T = ref->shape[1];
    int head_size = ctx->head_size;

    for (int h_idx = 0; h_idx < ctx->n_heads; ++h_idx) {
        Tensor* h_out = ctx->head_outputs[h_idx];
        for (int b = 0; b < B; ++b) {
            for (int t_ = 0; t_ < T; ++t_) {
                for (int d = 0; d < head_size; ++d) {
                    int out_d = h_idx * head_size + d;
                    h_out->grad[tensor_index3(h_out, b, t_, d)] += concat->grad[tensor_index3(concat, b, t_, out_d)];
                }
            }
        }
    }

    if (ctx->tracker == NULL) {
        for (int h_idx = 0; h_idx < ctx->n_heads; ++h_idx) {
            tensor_free(ctx->head_outputs[h_idx]);
            free(ctx->head_outputs[h_idx]);
        }
    }
    free(ctx->head_outputs);
    free(ctx);
}


/*
  Initialize MultiHeadAttention.

  Arguments:
    mha        : pointer to MultiHeadAttention
    embed_dim  : n_embd
    n_heads    : number of heads (must divide embed_dim)
    dropout_p  : dropout probability (ignored for now)
    causal     : whether to use causal masking in each head
*/
static inline void mha_init(MultiHeadAttention* mha,
                            int embed_dim,
                            int n_heads,
                            float dropout_p,
                            int causal) {
    if (embed_dim % n_heads != 0) {
        printf("mha_init: ERROR: embed_dim (%d) not divisible by n_heads (%d)\n",
               embed_dim, n_heads);
        return;
    }

    mha->embed_dim = embed_dim;
    mha->n_heads   = n_heads;
    mha->head_size = embed_dim / n_heads;
    mha->dropout_p = dropout_p;

    // Allocate array of heads
    mha->heads = (Head*)malloc(n_heads * sizeof(Head));
    if (mha->heads == NULL) {
        printf("mha_init: ERROR: malloc for heads failed\n");
        return;
    }

    // Initialize each head
    for (int i = 0; i < n_heads; ++i) {
        head_init(&mha->heads[i], embed_dim, mha->head_size, causal);
    }

    // Final projection: (embed_dim -> embed_dim), with bias
    linear_init(&mha->proj, embed_dim, embed_dim, 1 /* use_bias */);
}

static inline void mha_set_distributed(MultiHeadAttention* mha, int rank, int world_size) {
    (void)rank;
    (void)world_size;
}


/*
  Free MHA resources.
*/
static inline void mha_free(MultiHeadAttention* mha) {
    if (mha->heads != NULL) {
        for (int i = 0; i < mha->n_heads; ++i) {
            head_free(&mha->heads[i]);
        }
        free(mha->heads);
        mha->heads = NULL;
    }

    linear_free(&mha->proj);
}

static inline void mha_collect_params(MultiHeadAttention* mha, TensorPtrArray* list) {
    if (mha->heads != NULL) {
        for (int i = 0; i < mha->n_heads; ++i) {
            head_collect_params(&mha->heads[i], list);
        }
    }
    linear_collect_params(&mha->proj, list);
}

/*
  Forward pass:

  Input:
    x   : Tensor of shape (B, T, embed_dim)

  Output:
    out : Tensor of shape (B, T, embed_dim).

  Steps:
    1) For each head h:
         out_h = head_forward(h, x)            // (B, T, head_size)
    2) Concatenate out_h along last dim:
         concat[b,t,:] = [out_0, out_1, ..., out_{H-1}]   // (B, T, embed_dim)
    3) Apply final projection:
         out = proj(concat)                    // (B, T, embed_dim)
*/
static inline void mha_forward(MultiHeadAttention* mha,
                               const Tensor* x,
                               Tensor* out,
                               TensorTracker* tracker) {
    if (x->ndim != 3) {
        printf("mha_forward: ERROR: x->ndim must be 3 (B,T,C)\n");
        return;
    }

    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    if (C != mha->embed_dim) {
        printf("mha_forward: ERROR: input dim mismatch: C=%d, expected %d\n",
               C, mha->embed_dim);
        return;
    }

    // 1) Allocate concat tensor: (B, T, embed_dim)
    Tensor* concat = tensor_tracker_new(tracker);
    int concat_shape[3] = {B, T, mha->embed_dim};
    tensor_init(concat, 3, concat_shape);
    Tensor** head_outputs = (Tensor**)malloc(mha->n_heads * sizeof(Tensor*));

    // For each head, run head_forward and copy into concat
    for (int h_idx = 0; h_idx < mha->n_heads; ++h_idx) {
        head_outputs[h_idx] = tensor_tracker_new(tracker);
        head_forward(&mha->heads[h_idx], x, head_outputs[h_idx], tracker); // shape (B,T,head_size)

        // Copy into concat at offset [h_idx * head_size, (h_idx+1)*head_size)
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < mha->head_size; ++d) {
                    float v = tensor_get3(head_outputs[h_idx], b, t, d);
                    int out_d = h_idx * mha->head_size + d;
                    tensor_set3(concat, b, t, out_d, v);
                }
            }
        }
    }

    // Create context for concat so gradients reach individual heads
    ConcatHeadsContext* concat_ctx = (ConcatHeadsContext*)malloc(sizeof(ConcatHeadsContext));
    concat_ctx->tracker = tracker;
    concat_ctx->n_heads = mha->n_heads;
    concat_ctx->head_size = mha->head_size;
    concat_ctx->head_outputs = head_outputs;
    tensor_set_inputs(concat, head_outputs, mha->n_heads);
    concat->_ctx = concat_ctx;
    concat->_backward = concat_heads_backward;

    // 2) Apply final linear projection: proj(concat) -> out
    linear_forward(&mha->proj, concat, out);
}
