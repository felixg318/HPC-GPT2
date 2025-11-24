// block.h
// One Transformer block: LN -> MHA -> residual, LN -> MLP -> residual.

#pragma once

#include <stdio.h>
#include "tensor.h"
#include "layernorm.h"
#include "multihead_attention.h"
#include "mlp.h"
#include "add.h"

typedef struct {
    int embed_dim;      // n_embd

    LayerNorm ln1;      // first LayerNorm
    LayerNorm ln2;      // second LayerNorm

    MultiHeadAttention mha;  // multi-head self-attention
    MLP mlp;                  // feed-forward network
} Block;


/*
  Initialize a Block.

  Arguments:
    blk            : pointer to Block
    embed_dim      : n_embd
    n_heads        : number of attention heads
    attn_dropout_p : dropout for attention (ignored for now)
    mlp_dropout_p  : dropout for MLP (ignored for now)
    causal         : whether attention is causal (1) or not (0)
*/
static inline void block_init(Block* blk,
                              int embed_dim,
                              int n_heads,
                              float attn_dropout_p,
                              float mlp_dropout_p,
                              int causal) {
    blk->embed_dim = embed_dim;

    // LayerNorm with eps = 1e-5 (like PyTorch default)
    layernorm_init(&blk->ln1, embed_dim, 1e-5f);
    layernorm_init(&blk->ln2, embed_dim, 1e-5f);

    // Multi-head attention
    mha_init(&blk->mha, embed_dim, n_heads, attn_dropout_p, causal);

    // MLP
    mlp_init(&blk->mlp, embed_dim, mlp_dropout_p);
}


/*
  Free resources in a Block.
*/
static inline void block_free(Block* blk) {
    layernorm_free(&blk->ln1);
    layernorm_free(&blk->ln2);
    mha_free(&blk->mha);
    mlp_free(&blk->mlp);
}


/*
  Forward pass through one Transformer block.

  Input:
    x   : Tensor of shape (B, T, embed_dim)

  Output:
    y   : Tensor of shape (B, T, embed_dim) (allocated here)

  Equivalent to PyTorch:

    x = x + attn( LN1(x) )
    x = x + mlp( LN2(x) )
*/
static inline void block_forward(Block* blk,
                                 const Tensor* x,
                                 Tensor* y,
                                 TensorTracker* tracker) {
    if (x->ndim != 3) {
        printf("block_forward: ERROR: x->ndim must be 3 (B,T,C)\n");
        return;
    }

    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    if (C != blk->embed_dim) {
        printf("block_forward: ERROR: embed dim mismatch: C=%d, expected %d\n",
               C, blk->embed_dim);
        return;
    }

    // ----- First sub-layer: LN1 -> MHA -> residual -----

    Tensor* x_ln1 = tensor_tracker_new(tracker);       // LN1(x)
    layernorm_forward(&blk->ln1, x, x_ln1);  // shape (B,T,C)

    Tensor* attn_out = tensor_tracker_new(tracker);    // mha( LN1(x) )
    mha_forward(&blk->mha, x_ln1, attn_out, tracker); // shape (B,T,C)

    Tensor* x_res1 = tensor_tracker_new(tracker);      // x + attn_out
    add_forward(x, attn_out, x_res1);

    // ----- Second sub-layer: LN2 -> MLP -> residual -----

    Tensor* x_ln2 = tensor_tracker_new(tracker);       // LN2(x_res1)
    layernorm_forward(&blk->ln2, x_res1, x_ln2);

    Tensor* mlp_out = tensor_tracker_new(tracker);     // mlp( LN2(x_res1) )
    mlp_forward(&blk->mlp, x_ln2, mlp_out, tracker);  // shape (B,T,C)

    // y = x_res1 + mlp_out
    add_forward(x_res1, mlp_out, y);

    tensor_tracker_release(tracker, x_ln1);
    tensor_tracker_release(tracker, attn_out);
    tensor_tracker_release(tracker, x_res1);
    tensor_tracker_release(tracker, x_ln2);
    tensor_tracker_release(tracker, mlp_out);

    // 'y' now has shape (B,T,C) and must be freed by caller with tensor_free(y).
}
