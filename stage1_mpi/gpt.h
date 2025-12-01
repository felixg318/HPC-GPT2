// gpt.h
// Top-level GPT-2 style language model in C-style, with optional loss.

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "embedding.h"
#include "block.h"
#include "layernorm.h"
#include "linear.h"
#include "cross_entropy.h"
#include "broadcast.h"
#include "matmul.h"

typedef struct {
    int vocab_size;
    int block_size;
    int n_layer;
    int n_head;
    int n_embd;

    Embedding wte;     // token embeddings: (vocab_size, n_embd)
    Embedding wpe;     // positional embeddings: (block_size, n_embd)

    Block* blocks;     // array of length n_layer
    LayerNorm ln_f;    // final layer norm

    Linear lm_head;    // language modeling head: (n_embd -> vocab_size)
    TensorTracker activations; // track intermediate tensors for backward
} GPT;


/*
   Initialize GPT model.

   Same as before: no fancy init; you'll load real weights later.
*/
static inline void gpt_init(GPT* g,
                            int vocab_size,
                            int block_size,
                            int n_layer,
                            int n_head,
                            int n_embd,
                            float dropout_p) {
    g->vocab_size = vocab_size;
    g->block_size = block_size;
    g->n_layer    = n_layer;
    g->n_head     = n_head;
    g->n_embd     = n_embd;

    // Embeddings
    embedding_init(&g->wte, vocab_size, n_embd);
    embedding_init(&g->wpe, block_size, n_embd);

    // Blocks
    g->blocks = (Block*)malloc(n_layer * sizeof(Block));
    if (g->blocks == NULL) {
        printf("gpt_init: ERROR: malloc for blocks failed\n");
        return;
    }

    for (int i = 0; i < n_layer; ++i) {
        block_init(&g->blocks[i],
                   n_embd,
                   n_head,
                   dropout_p,   // attn dropout (ignored in forward)
                   dropout_p,   // mlp dropout (ignored in forward)
                   1            // causal attention
        );
    }

    // Final LayerNorm
    layernorm_init(&g->ln_f, n_embd, 1e-5f);

    // LM head: (n_embd -> vocab_size), no bias
    linear_init(&g->lm_head, n_embd, vocab_size, 0 /* use_bias=0 */);

    tensor_tracker_init(&g->activations);
}

static inline void gpt_set_distributed(GPT* g, int rank, int world_size) {
    if (g->blocks == NULL) return;
    for (int i = 0; i < g->n_layer; ++i) {
        block_set_distributed(&g->blocks[i], rank, world_size);
    }
    matmul_set_distributed(rank, world_size);
    softmax_set_distributed(rank, world_size);
    gelu_set_distributed(rank, world_size);
}


/*
   Free GPT resources.
*/
static inline void gpt_free(GPT* g) {
    embedding_free(&g->wte);
    embedding_free(&g->wpe);

    if (g->blocks != NULL) {
        for (int i = 0; i < g->n_layer; ++i) {
            block_free(&g->blocks[i]);
        }
        free(g->blocks);
        g->blocks = NULL;
    }

    layernorm_free(&g->ln_f);
    linear_free(&g->lm_head);

    tensor_tracker_free(&g->activations);
}

static inline void gpt_clear_activations(GPT* g) {
    tensor_tracker_reset(&g->activations);
}

static inline void gpt_collect_params(GPT* g, TensorPtrArray* list) {
    embedding_collect_params(&g->wte, list);
    embedding_collect_params(&g->wpe, list);
    for (int i = 0; i < g->n_layer; ++i) {
        block_collect_params(&g->blocks[i], list);
    }
    layernorm_collect_params(&g->ln_f, list);
    linear_collect_params(&g->lm_head, list);
}


/*
   Core forward that produces logits only (no loss).

   Input:
     idx    : int[B*T], token ids
     B, T   : batch size and sequence length

   Output:
     logits : Tensor of shape (B, T, vocab_size) (allocated here)
*/
static inline void gpt_forward_logits(GPT* g,
                                      const int* idx,
                                      int B,
                                      int T,
                                      Tensor* logits) {
    if (T > g->block_size) {
        printf("gpt_forward_logits: ERROR: T=%d exceeds block_size=%d\n", T, g->block_size);
        return;
    }

    TensorTracker* tracker = &g->activations;
    tensor_tracker_reset(tracker);

    // 1) Token & positional embeddings
    Tensor* tok_emb = tensor_tracker_new(tracker);
    embedding_forward_2d(&g->wte, idx, B, T, tok_emb);  // (B,T,n_embd)

    int* pos_idx = (int*)malloc(T * sizeof(int));
    if (pos_idx == NULL) {
        printf("gpt_forward_logits: ERROR: malloc for pos_idx failed\n");
        return;
    }
    for (int t = 0; t < T; ++t) pos_idx[t] = t;

    Tensor* pos_emb = tensor_tracker_new(tracker);
    embedding_forward_1d(&g->wpe, pos_idx, T, pos_emb); // (T,n_embd)
    free(pos_idx);

    // x = tok_emb + pos_emb
    Tensor* x = tensor_tracker_new(tracker);
    broadcast_add_forward(tok_emb, pos_emb, x);

    // 2) Blocks
    Tensor* current = x;

    for (int i = 0; i < g->n_layer; ++i) {
        Tensor* next = tensor_tracker_new(tracker);
        block_forward(&g->blocks[i], current, next, tracker);
        current = next;
    }

    // 3) Final LN
    Tensor* x_ln = tensor_tracker_new(tracker);
    layernorm_forward(&g->ln_f, current, x_ln);

    // 4) LM head
    linear_forward(&g->lm_head, x_ln, logits);
}


/*
   Forward with loss, like PyTorch:

     logits, loss = model(idx, targets)

   Inputs:
     idx     : int[B*T], token IDs
     targets : int[B*T], target token IDs
     B, T    : batch size, sequence length

   Outputs:
     logits_out : Tensor (B, T, vocab_size)
     loss_out   : Tensor of shape (1,)
*/
static inline void gpt_forward_with_loss(GPT* g,
                                         const int* idx,
                                         const int* targets,
                                         int B,
                                         int T,
                                         Tensor* logits_out,
                                         Tensor* loss_out) {
    // First compute logits
    gpt_forward_logits(g, idx, B, T, logits_out);

    // Then compute CE loss over all positions
    cross_entropy_loss_3d(logits_out, targets, B, T, loss_out);
}
