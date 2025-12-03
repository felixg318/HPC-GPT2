// multihead_attention.h
// Multi-head self-attention with optional head sharding across MPI ranks.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "tensor.h"
#include "linear.h"
#include "head.h"

typedef struct {
    int n_heads;       // number of heads
    int embed_dim;     // model dimension (n_embd)
    int head_size;     // embed_dim / n_heads
    float dropout_p;   // not used yet (inference only)

    int rank;
    int world_size;
    int local_head_start;
    int local_head_count;
    int* head_counts_per_rank;
    int* head_offsets_per_rank;

    Head* heads;       // array of length n_heads (weights remain replicated)
    Linear proj;       // final projection: (embed_dim -> embed_dim)
} MultiHeadAttention;

// Context for the multi-head attention operation
typedef struct {
    TensorTracker* tracker;
    int local_head_count;
    int head_offset;
    int head_size;
    Tensor** head_outputs;
} ConcatHeadsContext;

static inline void concat_heads_backward(Tensor* t) {
    ConcatHeadsContext* ctx = (ConcatHeadsContext*)t->_ctx;
    if (ctx == NULL) return;

    if (ctx->local_head_count > 0) {
        Tensor* ref = ctx->head_outputs[0];
        int B = ref->shape[0];
        int T = ref->shape[1];
        int head_size = ctx->head_size;
        int head_offset = ctx->head_offset;

        for (int local_idx = 0; local_idx < ctx->local_head_count; ++local_idx) {
            Tensor* h_out = ctx->head_outputs[local_idx];
            int global_idx = head_offset + local_idx;
            for (int b = 0; b < B; ++b) {
                for (int t_ = 0; t_ < T; ++t_) {
                    for (int d = 0; d < head_size; ++d) {
                        int out_d = global_idx * head_size + d;
                        h_out->grad[tensor_index3(h_out, b, t_, d)] +=
                            t->grad[tensor_index3(t, b, t_, out_d)];
                    }
                }
            }
        }

        if (ctx->tracker == NULL) {
            for (int i = 0; i < ctx->local_head_count; ++i) {
                tensor_free(ctx->head_outputs[i]);
                free(ctx->head_outputs[i]);
            }
        }
        free(ctx->head_outputs);
    } else if (ctx->head_outputs != NULL) {
        free(ctx->head_outputs);
    }
    free(ctx);
}

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
    mha->rank = 0;
    mha->world_size = 1;
    mha->local_head_start = 0;
    mha->local_head_count = n_heads;
    mha->head_counts_per_rank = NULL;
    mha->head_offsets_per_rank = NULL;

    mha->heads = (Head*)malloc((size_t)n_heads * sizeof(Head));
    if (mha->heads == NULL) {
        printf("mha_init: ERROR: malloc for heads failed\n");
        return;
    }
    for (int i = 0; i < n_heads; ++i) {
        head_init(&mha->heads[i], embed_dim, mha->head_size, causal);
    }

    linear_init(&mha->proj, embed_dim, embed_dim, 1 /* use_bias */);
}

static inline void mha_set_distributed(MultiHeadAttention* mha, int rank, int world_size) {
    if (mha == NULL) return;
    mha->rank = (rank >= 0) ? rank : 0;
    mha->world_size = (world_size > 0) ? world_size : 1;
    int base = (mha->world_size > 0) ? (mha->n_heads / mha->world_size) : mha->n_heads;
    int extra = (mha->world_size > 0) ? (mha->n_heads % mha->world_size) : 0;
    if (mha->head_counts_per_rank != NULL) {
        free(mha->head_counts_per_rank);
        mha->head_counts_per_rank = NULL;
    }
    if (mha->head_offsets_per_rank != NULL) {
        free(mha->head_offsets_per_rank);
        mha->head_offsets_per_rank = NULL;
    }
    mha->head_counts_per_rank = (int*)malloc((size_t)mha->world_size * sizeof(int));
    mha->head_offsets_per_rank = (int*)malloc((size_t)mha->world_size * sizeof(int));
    if (mha->head_counts_per_rank == NULL || mha->head_offsets_per_rank == NULL) {
        free(mha->head_counts_per_rank);
        free(mha->head_offsets_per_rank);
        mha->head_counts_per_rank = NULL;
        mha->head_offsets_per_rank = NULL;
        printf("mha_set_distributed: failed to allocate rank metadata\n");
        return;
    }
    int offset = 0;
    for (int r = 0; r < mha->world_size; ++r) {
        int count = base + (r < extra ? 1 : 0);
        mha->head_counts_per_rank[r] = count;
        mha->head_offsets_per_rank[r] = offset;
        if (r == mha->rank) {
            mha->local_head_start = offset;
            mha->local_head_count = count;
        }
        offset += count;
    }
}

static inline void mha_free(MultiHeadAttention* mha) {
    if (mha->heads != NULL) {
        for (int i = 0; i < mha->n_heads; ++i) {
            head_free(&mha->heads[i]);
        }
        free(mha->heads);
        mha->heads = NULL;
    }
    linear_free(&mha->proj);
    if (mha->head_counts_per_rank != NULL) {
        free(mha->head_counts_per_rank);
        mha->head_counts_per_rank = NULL;
    }
    if (mha->head_offsets_per_rank != NULL) {
        free(mha->head_offsets_per_rank);
        mha->head_offsets_per_rank = NULL;
    }
}

static inline void mha_collect_params(MultiHeadAttention* mha, TensorPtrArray* list) {
    if (mha == NULL) return;
    if (mha->heads != NULL) {
        for (int i = 0; i < mha->n_heads; ++i) {
            head_collect_params(&mha->heads[i], list);
        }
    }
    linear_collect_params(&mha->proj, list);
}

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

    Tensor* concat = tensor_tracker_new(tracker);
    int concat_shape[3] = {B, T, mha->embed_dim};
    tensor_init(concat, 3, concat_shape);

    int local_heads = mha->local_head_count;
    int head_start = mha->local_head_start;
    Tensor** head_outputs = NULL;
    if (local_heads > 0) {
        head_outputs = (Tensor**)malloc((size_t)local_heads * sizeof(Tensor*));
        if (head_outputs == NULL) {
            printf("mha_forward: failed to allocate head_outputs\n");
            return;
        }
    }

    for (int local_idx = 0; local_idx < local_heads; ++local_idx) {
        int global_idx = head_start + local_idx;
        head_outputs[local_idx] = tensor_tracker_new(tracker);
        head_forward(&mha->heads[global_idx], x, head_outputs[local_idx], tracker);
    }

    int local_dim = local_heads * mha->head_size;
    size_t local_elems = (size_t)B * T * local_dim;
    float* local_concat = NULL;
    if (local_elems > 0) {
        local_concat = (float*)malloc(local_elems * sizeof(float));
        if (local_concat == NULL) {
            printf("mha_forward: failed to allocate local concat buffer\n");
        } else {
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    for (int local_idx = 0; local_idx < local_heads; ++local_idx) {
                        for (int d = 0; d < mha->head_size; ++d) {
                            size_t col = (size_t)local_idx * (size_t)mha->head_size + (size_t)d;
                            size_t idx = ((size_t)b * T + (size_t)t) * (size_t)local_dim + col;
                            local_concat[idx] = tensor_get3(head_outputs[local_idx], b, t, d);
                        }
                    }
                }
            }
        }
    }

    if (mha->world_size > 1 && concat->data != NULL) {
        int* recv_counts = (int*)malloc((size_t)mha->world_size * sizeof(int));
        int* displs = (int*)malloc((size_t)mha->world_size * sizeof(int));
        if (recv_counts == NULL || displs == NULL) {
            printf("mha_forward: failed to allocate recv counts\n");
        } else {
            int bt = B * T;
            for (int r = 0; r < mha->world_size; ++r) {
                int r_heads = (mha->head_counts_per_rank != NULL) ? mha->head_counts_per_rank[r] : 0;
                int r_offset = (mha->head_offsets_per_rank != NULL) ? mha->head_offsets_per_rank[r] : 0;
                recv_counts[r] = bt * r_heads * mha->head_size;
                displs[r] = bt * r_offset * mha->head_size;
            }
            float* send_ptr = (local_concat != NULL) ? local_concat : concat->data;
            MPI_Allgatherv(send_ptr,
                           (int)local_elems,
                           MPI_FLOAT,
                           concat->data,
                           recv_counts,
                           displs,
                           MPI_FLOAT,
                           MPI_COMM_WORLD);
        }
        free(recv_counts);
        free(displs);
    } else {
        for (int local_idx = 0; local_idx < local_heads; ++local_idx) {
            int global_idx = head_start + local_idx;
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < T; ++t) {
                    for (int d = 0; d < mha->head_size; ++d) {
                        float v = tensor_get3(head_outputs[local_idx], b, t, d);
                        int out_d = global_idx * mha->head_size + d;
                        tensor_set3(concat, b, t, out_d, v);
                    }
                }
            }
        }
    }
    if (local_concat != NULL) {
        free(local_concat);
    }

    ConcatHeadsContext* concat_ctx = (ConcatHeadsContext*)malloc(sizeof(ConcatHeadsContext));
    concat_ctx->tracker = tracker;
    concat_ctx->local_head_count = local_heads;
    concat_ctx->head_offset = head_start;
    concat_ctx->head_size = mha->head_size;
    concat_ctx->head_outputs = head_outputs;
    if (local_heads > 0) {
        tensor_set_inputs(concat, head_outputs, local_heads);
    } else {
        tensor_set_inputs(concat, NULL, 0);
    }
    concat->_ctx = concat_ctx;
    concat->_backward = concat_heads_backward;

    linear_forward(&mha->proj, concat, out);
}
