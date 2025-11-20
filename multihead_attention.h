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
static inline void mha_forward(MultiHeadAttention* mha, const Tensor* x, Tensor* out) {
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
    Tensor concat;
    int concat_shape[3] = {B, T, mha->embed_dim};
    tensor_init(&concat, 3, concat_shape);

    // For each head, run head_forward and copy into concat
    for (int h_idx = 0; h_idx < mha->n_heads; ++h_idx) {
        Tensor h_out;
        head_forward(&mha->heads[h_idx], x, &h_out); // shape (B,T,head_size)

        // Copy into concat at offset [h_idx * head_size, (h_idx+1)*head_size)
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < mha->head_size; ++d) {
                    float v = tensor_get3(&h_out, b, t, d);
                    int out_d = h_idx * mha->head_size + d;
                    tensor_set3(&concat, b, t, out_d, v);
                }
            }
        }

        // Free h_out after copying
        tensor_free(&h_out);
    }

    // 2) Apply final linear projection: proj(concat) -> out
    linear_forward(&mha->proj, &concat, out);

    // Free intermediate concat
    tensor_free(&concat);

    // NOTE: out is allocated inside linear_forward; caller must tensor_free(out).
}

