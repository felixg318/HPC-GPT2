// dataloader.h
// Dataloader for text data.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "tokenizer.h"
#include "tensor.h"

typedef struct {
    int* tokens;
    int num_tokens;
    int batch_size;
    int seq_len;
    int current_pos;
    int rank;
    int world_size;
} DataLoader;

static inline void dataloader_init_with_tokenizer(DataLoader* dl,
                                                  const Tokenizer* tokenizer,
                                                  int batch_size,
                                                  int seq_len) {
    dl->num_tokens = tokenizer_data_len(tokenizer);
    dl->tokens = (int*)malloc(dl->num_tokens * sizeof(int));
    if (dl->tokens == NULL) {
        printf("dataloader_init_with_tokenizer: ERROR: malloc failed\n");
        dl->num_tokens = 0;
        dl->batch_size = batch_size;
        dl->seq_len = seq_len;
        dl->current_pos = 0;
        return;
    }
    memcpy(dl->tokens, tokenizer_data_ptr(tokenizer), dl->num_tokens * sizeof(int));
    
    dl->batch_size = batch_size;
    dl->seq_len = seq_len;
    dl->current_pos = 0;
    dl->rank = 0;
    dl->world_size = 1;
}

static inline void dataloader_free(DataLoader* dl) {
    if (dl->tokens != NULL) {
        free(dl->tokens);
        dl->tokens = NULL;
    }
    dl->num_tokens = 0;
}

static inline void dataloader_set_distributed(DataLoader* dl, int rank, int world_size) {
    if (dl == NULL) return;
    dl->rank = rank;
    dl->world_size = (world_size > 0) ? world_size : 1;
}

/*
  Broadcast dataloader token buffer and metadata from root to all ranks.
*/
static inline int dataloader_broadcast(DataLoader* dl, int root_rank) {
    if (dl == NULL) return 0;
    int meta[3];
    if (dl->rank == root_rank) {
        meta[0] = dl->num_tokens;
        meta[1] = dl->batch_size;
        meta[2] = dl->seq_len;
    }
    MPI_Bcast(meta, 3, MPI_INT, root_rank, MPI_COMM_WORLD);
    int num_tokens = meta[0];
    int batch_size = meta[1];
    int seq_len = meta[2];

    if (dl->rank != root_rank) {
        dl->num_tokens = num_tokens;
        dl->batch_size = batch_size;
        dl->seq_len = seq_len;
        dl->current_pos = 0;
    }

    int ok = 1;
    if (num_tokens > 0) {
        if (dl->rank != root_rank || dl->tokens == NULL) {
            int* buf = (int*)malloc(num_tokens * sizeof(int));
            if (buf == NULL) {
                ok = 0;
            } else {
                dl->tokens = buf;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) {
            if (dl->rank != root_rank && dl->tokens != NULL) {
                free(dl->tokens);
                dl->tokens = NULL;
            }
            return 0;
        }
        MPI_Bcast(dl->tokens, num_tokens, MPI_INT, root_rank, MPI_COMM_WORLD);
    }
    return 1;
}

static inline void dataloader_next_batch(DataLoader* dl, int** inputs, int** targets) {
    int batch_size = dl->batch_size;
    int seq_len = dl->seq_len;
    int world_size = dl->world_size > 0 ? dl->world_size : 1;
    if (dl->num_tokens <= 1) {
        printf("dataloader_next_batch: insufficient tokens\n");
        *inputs = NULL;
        *targets = NULL;
        return;
    }
    
    *inputs = (int*)malloc(batch_size * seq_len * sizeof(int));
    *targets = (int*)malloc(batch_size * seq_len * sizeof(int));
    
    size_t tokens_total = (size_t)dl->num_tokens;
    size_t rank_offset = (size_t)dl->rank * batch_size * seq_len;
    size_t base_pos = (size_t)dl->current_pos + rank_offset;

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            size_t token_pos = base_pos + (size_t)b * seq_len + (size_t)t;
            size_t next_pos = token_pos + 1;
            int input_idx = (int)(token_pos % tokens_total);
            int target_idx = (int)(next_pos % tokens_total);
            (*inputs)[b * seq_len + t] = dl->tokens[input_idx];
            (*targets)[b * seq_len + t] = dl->tokens[target_idx];
        }
    }
    
    int stride = batch_size * seq_len * world_size;
    dl->current_pos += stride;
    if (dl->current_pos + stride + 1 >= dl->num_tokens) {
        // Advance start position using the same RNG seeded via tensor_set_seed for reproducibility.
        int offset = (int)(tensor_rand_uniform() * (float)dl->num_tokens);
        if (offset >= dl->num_tokens) offset = dl->num_tokens - 1;
        dl->current_pos = offset;
    }
}
