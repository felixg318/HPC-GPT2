// dataloader.h
// Dataloader for text data.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

typedef struct {
    int* tokens;
    int num_tokens;
    int batch_size;
    int seq_len;
    int current_pos;
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
}

static inline void dataloader_free(DataLoader* dl) {
    if (dl->tokens != NULL) {
        free(dl->tokens);
        dl->tokens = NULL;
    }
    dl->num_tokens = 0;
}

static inline void dataloader_next_batch(DataLoader* dl, int** inputs, int** targets) {
    int batch_size = dl->batch_size;
    int seq_len = dl->seq_len;
    
    *inputs = (int*)malloc(batch_size * seq_len * sizeof(int));
    *targets = (int*)malloc(batch_size * seq_len * sizeof(int));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int token_pos = dl->current_pos + b * seq_len + t;
            if (token_pos + 1 < dl->num_tokens) {
                (*inputs)[b * seq_len + t] = dl->tokens[token_pos];
                (*targets)[b * seq_len + t] = dl->tokens[token_pos + 1];
            } else {
                // End of data, wrap around
                (*inputs)[b * seq_len + t] = dl->tokens[token_pos % dl->num_tokens];
                (*targets)[b * seq_len + t] = dl->tokens[(token_pos + 1) % dl->num_tokens];
            }
        }
    }
    
    dl->current_pos += batch_size * seq_len;
    if (dl->current_pos + batch_size * seq_len + 1 >= dl->num_tokens) {
        dl->current_pos = 0;
    }
}
