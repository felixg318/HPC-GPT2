// head.h
// One head of causal self-attention.
// Input : x (B, T, C_in = embed_dim)
// Output: out (B, T, head_size)

#pragma once

#include <math.h>
#include <stdio.h>

#include "tensor.h"
#include "linear.h"
#include "softmax.h"

typedef struct {
    int head_size;   // dimension of this head (hs)
    int embed_dim;   // input dimension (C_in = n_embd)
    int causal;      // 1 = apply causal mask, 0 = no mask

    Linear key;      // (embed_dim -> head_size)
    Linear query;    // (embed_dim -> head_size)
    Linear value;    // (embed_dim -> head_size)
} Head;


/*
  Initialize a Head.

  Arguments:
    h          : pointer to Head struct
    embed_dim  : input dimension (n_embd)
    head_size  : dimension of this head (n_embd / n_head)
    causal     : 1 to enable causal mask (only attend to t' <= t)
*/
static inline void head_init(Head* h, int embed_dim, int head_size, int causal) {
    h->embed_dim = embed_dim;
    h->head_size = head_size;
    h->causal    = causal;

    // In PyTorch Head, bias=False for q,k,v
    linear_init(&h->key,   embed_dim, head_size, 0 /* use_bias = 0 */);
    linear_init(&h->query, embed_dim, head_size, 0);
    linear_init(&h->value, embed_dim, head_size, 0);
}


/*
  Free memory inside Head.
*/
static inline void head_free(Head* h) {
    linear_free(&h->key);
    linear_free(&h->query);
    linear_free(&h->value);
}


/*
  Forward pass for one attention head.

  Input:
    x    : Tensor with shape (B, T, embed_dim)
  Output:
    out  : Tensor with shape (B, T, head_size)

  Steps:
    1) q = x * W_q
       k = x * W_k
       v = x * W_v
    2) att[b, t_q, t_k] = (q[b,t_q,:] · k[b,t_k,:]) / sqrt(head_size)
       if causal && t_k > t_q: set att to very negative number
    3) softmax over last dim (t_k)
    4) out[b, t_q, d] = sum_{t_k} att[b, t_q, t_k] * v[b, t_k, d]
*/
static inline void head_forward(const Head* h, const Tensor* x, Tensor* out) {
    if (x->ndim != 3) {
        printf("head_forward: ERROR: x->ndim must be 3 (B,T,C)\n");
        return;
    }

    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    if (C != h->embed_dim) {
        printf("head_forward: ERROR: input dim mismatch: C=%d, expected %d\n",
               C, h->embed_dim);
        return;
    }

    // 1) Compute q, k, v: each has shape (B, T, head_size)
    Tensor q, k, v;
    linear_forward(&h->query, x, &q);
    linear_forward(&h->key,   x, &k);
    linear_forward(&h->value, x, &v);

    // 2) Compute attention logits: att[b, t_q, t_k]
    // Shape: (B, T, T)
    Tensor att;
    int att_shape[3] = {B, T, T};
    tensor_init(&att, 3, att_shape);

    float scale = 1.0f / sqrtf((float)h->head_size);

    for (int b = 0; b < B; ++b) {
        for (int t_q = 0; t_q < T; ++t_q) {
            for (int t_k = 0; t_k < T; ++t_k) {

                // Dot product over head_size
                float dot = 0.0f;
                for (int d = 0; d < h->head_size; ++d) {
                    float q_bd = tensor_get3(&q, b, t_q, d);
                    float k_bd = tensor_get3(&k, b, t_k, d);
                    dot += q_bd * k_bd;
                }

                float score = dot * scale;

                // Causal mask: disallow attending to future positions
                if (h->causal && t_k > t_q) {
                    score = -1e30f;  // very negative -> softmax ~ 0
                }

                tensor_set3(&att, b, t_q, t_k, score);
            }
        }
    }

    // 3) Softmax over last dimension (t_k)
    Tensor att_probs;
    softmax_forward(&att, &att_probs);  // same shape (B, T, T)

    // No longer need att
    tensor_free(&att);

    // 4) Weighted sum of values v: out[b, t_q, d] = sum_{t_k} att[b,t_q,t_k]*v[b,t_k,d]
    int out_shape[3] = {B, T, h->head_size};
    tensor_init(out, 3, out_shape);

    for (int b = 0; b < B; ++b) {
        for (int t_q = 0; t_q < T; ++t_q) {
            for (int d = 0; d < h->head_size; ++d) {
                float sum = 0.0f;

                for (int t_k = 0; t_k < T; ++t_k) {
                    float alpha = tensor_get3(&att_probs, b, t_q, t_k);  // attention weight
                    float v_val = tensor_get3(&v, b, t_k, d);           // value
                    sum += alpha * v_val;
                }

                tensor_set3(out, b, t_q, d, sum);
            }
        }
    }

    // Free temporaries
    tensor_free(&q);
    tensor_free(&k);
    tensor_free(&v);
    tensor_free(&att_probs);
}

