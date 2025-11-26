// embedding.h
// Simple embedding layer: looks up rows from a (num_embeddings, embedding_dim) matrix.

#pragma once

#include <stdio.h>
#include "tensor.h"

/*
   Embedding struct:

   - num_embeddings : vocabulary size (for tokens) or max positions (for positions)
   - embedding_dim  : dimension of each embedding vector (n_embd)
   - weight         : Tensor of shape (num_embeddings, embedding_dim)

   PyTorch analogy: nn.Embedding(num_embeddings, embedding_dim)
*/
typedef struct {
    int num_embeddings;
    int embedding_dim;
    Tensor weight;   // shape: (num_embeddings, embedding_dim)
} Embedding;

// Context for the embedding operation
typedef struct {
    Embedding* emb;
    const int* idx;
    int B;
    int T;
} EmbeddingContext;


/*
  Initialize an Embedding.

  Arguments:
    emb            : pointer to Embedding
    num_embeddings : number of rows (e.g., vocab_size or block_size)
    embedding_dim  : size of each embedding vector (n_embd)
*/
static inline void embedding_init(Embedding* emb, int num_embeddings, int embedding_dim) {
    emb->num_embeddings = num_embeddings;
    emb->embedding_dim  = embedding_dim;

    int w_shape[2];
    w_shape[0] = num_embeddings;
    w_shape[1] = embedding_dim;
    tensor_init(&emb->weight, 2, w_shape);  // (num_embeddings, embedding_dim)

    // For now, we can fill with zeros or some small constant.
    // Later we will load real weights from PyTorch.
    tensor_fill(&emb->weight, 0.0f);
}

/*
  Free embedding memory.
*/
static inline void embedding_free(Embedding* emb) {
    tensor_free(&emb->weight);
}

static inline void embedding_collect_params(Embedding* emb, TensorPtrArray* list) {
    tensor_ptr_array_push(list, &emb->weight);
}

// Backward function for 2D embedding
static inline void embedding_backward_2d(Tensor* t) {
    EmbeddingContext* ctx = (EmbeddingContext*)t->_ctx;
    Embedding* emb = ctx->emb;
    const int* idx = ctx->idx;
    int B = ctx->B;
    int T = ctx->T;
    Tensor* out = t;

    for (int b = 0; b < B; ++b) {
        for (int t_ = 0; t_ < T; ++t_) {
            int token_id = idx[b * T + t_];
            if (token_id >= 0 && token_id < emb->num_embeddings) {
                for (int d = 0; d < emb->embedding_dim; ++d) {
                    emb->weight.grad[tensor_index2(&emb->weight, token_id, d)] += out->grad[tensor_index3(out, b, t_, d)];
                }
            }
        }
    }
    free(ctx);
}

// Backward function for 1D embedding
static inline void embedding_backward_1d(Tensor* t) {
    EmbeddingContext* ctx = (EmbeddingContext*)t->_ctx;
    Embedding* emb = ctx->emb;
    const int* idx = ctx->idx;
    int T = ctx->T;
    Tensor* out = t;

    for (int t_ = 0; t_ < T; ++t_) {
        int id = idx[t_];
        if (id >= 0 && id < emb->num_embeddings) {
            for (int d = 0; d < emb->embedding_dim; ++d) {
                emb->weight.grad[tensor_index2(&emb->weight, id, d)] += out->grad[tensor_index2(out, t_, d)];
            }
        }
    }
    free(ctx);
}


/*
  Forward pass for token indices, 2D case:

    idx: int array of shape (B, T)  (we just pass as int* with known B,T)
    out: Tensor, allocated here, shape (B, T, embedding_dim)

  out[b, t, :] = weight[ idx[b, t], : ]
*/
static inline void embedding_forward_2d(
    Embedding* emb,
    const int* idx,   // pointer to int[B*T]
    int B,
    int T,
    Tensor* out
) {
    int out_shape[3];
    out_shape[0] = B;
    out_shape[1] = T;
    out_shape[2] = emb->embedding_dim;
    tensor_init(out, 3, out_shape);  // (B, T, embedding_dim)

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int token_id = idx[b * T + t];  // row in weight

            if (token_id < 0 || token_id >= emb->num_embeddings) {
                printf("embedding_forward_2d: WARNING: token_id=%d out of range [0,%d)\n",
                       token_id, emb->num_embeddings);
                token_id = 0; // clamp or handle as you wish
            }

            // Copy embedding vector: weight[token_id, :] -> out[b, t, :]
            for (int d = 0; d < emb->embedding_dim; ++d) {
                float val = tensor_get2(&emb->weight, token_id, d);     // weight[token_id, d]
                tensor_set3(out, b, t, d, val);                         // out[b, t, d]
            }
        }
    }
    
    // Create context for autograd
    EmbeddingContext* ctx = (EmbeddingContext*)malloc(sizeof(EmbeddingContext));
    ctx->emb = emb;
    ctx->idx = idx;
    ctx->B = B;
    ctx->T = T;
    
    out->_ctx = ctx;
    out->_backward = embedding_backward_2d;
}


/*
  Forward for 1D indices (e.g. positions 0..T-1):

    idx: int array of shape (T)
    out: Tensor, shape (T, embedding_dim)

  out[t, :] = weight[ idx[t], : ]
*/
static inline void embedding_forward_1d(
    Embedding* emb,
    const int* idx,  // pointer to int[T]
    int T,
    Tensor* out
) {
    int out_shape[2];
    out_shape[0] = T;
    out_shape[1] = emb->embedding_dim;
    tensor_init(out, 2, out_shape);  // (T, embedding_dim)

    for (int t = 0; t < T; ++t) {
        int id = idx[t];

        if (id < 0 || id >= emb->num_embeddings) {
            printf("embedding_forward_1d: WARNING: id=%d out of range [0,%d)\n",
                   id, emb->num_embeddings);
            id = 0;
        }

        for (int d = 0; d < emb->embedding_dim; ++d) {
            float val = tensor_get2(&emb->weight, id, d);   // weight[id, d]
            tensor_set2(out, t, d, val);                    // out[t, d]
        }
    }
    
    // Create context for autograd
    EmbeddingContext* ctx = (EmbeddingContext*)malloc(sizeof(EmbeddingContext));
    ctx->emb = emb;
    ctx->idx = idx;
    ctx->B = 0;
    ctx->T = T;

    out->_ctx = ctx;
    out->_backward = embedding_backward_1d;
}
