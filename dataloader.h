// dataloader.h
// Dataloader for text data.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* text;
    int* tokens;
    int num_tokens;
    int batch_size;
    int seq_len;
    int current_pos;
} DataLoader;

// A simple whitespace tokenizer.
// In a real application, you would use a proper tokenizer like BPE.
// A simple character-to-integer mapping.
// This is a placeholder for a real tokenizer (e.g., BPE).
static inline int char_to_int(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - 'a';
    } else if (c == ' ') {
        return 26; // Map space to 26
    } else if (c == '.') {
        return 27; // Map period to 27
    } else if (c == '\n') {
        return 28; // Map newline to 28
    } else {
        return 26; // Default to space for unknown characters
    }
}

static inline void tokenize(char* text, int* tokens, int* num_tokens) {
    int count = 0;
    for (int i = 0; text[i] != '\0'; ++i) {
        tokens[count++] = char_to_int(text[i]);
    }
    *num_tokens = count;
}


static inline void dataloader_init(DataLoader* dl, const char* filename, int batch_size, int seq_len) {
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
        printf("dataloader_init: ERROR: could not open file %s\n", filename);
        return;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    dl->text = (char*)malloc(fsize + 1);
    fread(dl->text, 1, fsize, f);
    fclose(f);
    dl->text[fsize] = 0;
    
    dl->tokens = (int*)malloc(fsize * sizeof(int)); // oversized, but safe
    tokenize(dl->text, dl->tokens, &dl->num_tokens);
    
    dl->batch_size = batch_size;
    dl->seq_len = seq_len;
    dl->current_pos = 0;
}

static inline void dataloader_free(DataLoader* dl) {
    free(dl->text);
    free(dl->tokens);
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
