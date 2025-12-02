#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char* file_path;
    char* text;
    size_t text_len;

    char** token_list;
    int token_count;
    int token_capacity;

    char** vocab;
    int vocab_size;
    int vocab_capacity;

    int* encoded;
    int encoded_count;
    int encoded_capacity;

    int pad_id;
    int eos_id;
} Tokenizer;

static inline void tokenizer_init(Tokenizer* tok, const char* path) {
    tok->file_path = NULL;
    tok->text = NULL;
    tok->text_len = 0;
    tok->token_list = NULL;
    tok->token_count = 0;
    tok->token_capacity = 0;
    tok->vocab = NULL;
    tok->vocab_size = 0;
    tok->vocab_capacity = 0;
    tok->encoded = NULL;
    tok->encoded_count = 0;
    tok->encoded_capacity = 0;
    tok->pad_id = -1;
    tok->eos_id = -1;
    if (path != NULL) {
        size_t len = strlen(path);
        tok->file_path = (char*)malloc(len + 1);
        if (tok->file_path != NULL) {
            memcpy(tok->file_path, path, len + 1);
        }
    }
}

static inline void tokenizer_free(Tokenizer* tok) {
    if (tok->file_path) free(tok->file_path);
    if (tok->text) free(tok->text);
    for (int i = 0; i < tok->token_count; ++i) {
        free(tok->token_list[i]);
    }
    free(tok->token_list);
    for (int i = 0; i < tok->vocab_size; ++i) {
        free(tok->vocab[i]);
    }
    free(tok->vocab);
    free(tok->encoded);
    tokenizer_init(tok, NULL);
}

static inline int tokenizer_reserve_tokens(Tokenizer* tok, int capacity) {
    if (capacity <= tok->token_capacity) return 1;
    char** new_list = (char**)realloc(tok->token_list, capacity * sizeof(char*));
    if (!new_list) return 0;
    tok->token_list = new_list;
    tok->token_capacity = capacity;
    return 1;
}

static inline int tokenizer_reserve_vocab(Tokenizer* tok, int capacity) {
    if (capacity <= tok->vocab_capacity) return 1;
    char** new_vocab = (char**)realloc(tok->vocab, capacity * sizeof(char*));
    if (!new_vocab) return 0;
    tok->vocab = new_vocab;
    tok->vocab_capacity = capacity;
    return 1;
}

static inline int tokenizer_reserve_encoded(Tokenizer* tok, int capacity) {
    if (capacity <= tok->encoded_capacity) return 1;
    int* new_data = (int*)realloc(tok->encoded, capacity * sizeof(int));
    if (!new_data) return 0;
    tok->encoded = new_data;
    tok->encoded_capacity = capacity;
    return 1;
}

static inline int tokenizer_add_token_to_list(Tokenizer* tok, const char* start, size_t len) {
    if (len == 0) return 1;
    if (!tokenizer_reserve_tokens(tok, tok->token_count + 1)) return 0;
    char* copy = (char*)malloc(len + 1);
    if (!copy) return 0;
    memcpy(copy, start, len);
    copy[len] = '\0';
    tok->token_list[tok->token_count++] = copy;
    return 1;
}

static inline int tokenizer_read_file(Tokenizer* tok) {
    if (tok->file_path == NULL) {
        printf("tokenizer_read_file: no file specified\n");
        return 0;
    }
    FILE* f = fopen(tok->file_path, "rb");
    if (!f) {
        printf("tokenizer_read_file: failed to open %s\n", tok->file_path);
        return 0;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) {
        fclose(f);
        printf("tokenizer_read_file: empty file\n");
        return 0;
    }
    tok->text = (char*)malloc(fsize + 1);
    if (!tok->text) {
        fclose(f);
        printf("tokenizer_read_file: malloc failed\n");
        return 0;
    }
    size_t read = fread(tok->text, 1, fsize, f);
    fclose(f);
    tok->text[read] = '\0';
    tok->text_len = read;
    return 1;
}

static inline int tokenizer_extract(Tokenizer* tok) {
    if (!tokenizer_read_file(tok)) return 0;
    const char* ptr = tok->text;
    const char* start = NULL;
    size_t len = 0;
    for (size_t i = 0; i <= tok->text_len; ++i) {
        unsigned char ch = (i < tok->text_len) ? (unsigned char)ptr[i] : (unsigned char)' ';
        if (isalnum(ch)) {
            if (!start) start = &ptr[i];
            ++len;
        } else {
            if (start && len > 0) {
                if (!tokenizer_add_token_to_list(tok, start, len)) return 0;
                start = NULL;
                len = 0;
            }
            if (!isspace(ch) && ch != 0) {
                char c = (char)ch;
                if (!tokenizer_add_token_to_list(tok, &c, 1)) return 0;
            }
        }
    }
    return tok->token_count > 0;
}

static inline int tokenizer_find_vocab(Tokenizer* tok, const char* token) {
    for (int i = 0; i < tok->vocab_size; ++i) {
        if (strcmp(tok->vocab[i], token) == 0) {
            return i;
        }
    }
    return -1;
}

static inline char* tokenizer_strdup(const char* src) {
    if (src == NULL) return NULL;
    size_t len = strlen(src);
    char* copy = (char*)malloc(len + 1);
    if (copy != NULL) {
        memcpy(copy, src, len + 1);
    }
    return copy;
}

static inline int tokenizer_add_vocab(Tokenizer* tok, const char* token) {
    int existing = tokenizer_find_vocab(tok, token);
    if (existing >= 0) {
        return existing;
    }
    if (!tokenizer_reserve_vocab(tok, tok->vocab_size + 1)) return -1;
    char* copy = tokenizer_strdup(token);
    if (!copy) return -1;
    tok->vocab[tok->vocab_size] = copy;
    return tok->vocab_size++;
}

static inline void tokenizer_append_eos(Tokenizer* tok) {
    if (tok == NULL) return;
    if (tok->eos_id < 0) {
        int id = tokenizer_add_vocab(tok, "[EOS]");
        if (id < 0) return;
        tok->eos_id = id;
    }
    if (tok->encoded_count > 0 && tok->encoded[tok->encoded_count - 1] == tok->eos_id) {
        return;
    }
    if (!tokenizer_reserve_encoded(tok, tok->encoded_count + 1)) return;
    tok->encoded[tok->encoded_count++] = tok->eos_id;
}

static inline int tokenizer_encode(Tokenizer* tok) {
    tok->encoded_count = 0;
    for (int i = 0; i < tok->token_count; ++i) {
        const char* token = tok->token_list[i];
        int id = tokenizer_add_vocab(tok, token);
        if (id < 0) return 0;
        if (!tokenizer_reserve_encoded(tok, tok->encoded_count + 1)) return 0;
        tok->encoded[tok->encoded_count++] = id;
    }
    for (int i = 0; i < tok->token_count; ++i) {
        free(tok->token_list[i]);
    }
    tok->token_count = 0;
    if (tok->encoded_count > 0) {
        tokenizer_append_eos(tok);
    }
    return tok->encoded_count > 0;
}

static inline void tokenizer_pad_to(Tokenizer* tok, size_t new_size) {
    int removed_eos = 0;
    if (tok->eos_id >= 0 && tok->encoded_count > 0 &&
        tok->encoded[tok->encoded_count - 1] == tok->eos_id) {
        --tok->encoded_count;
        removed_eos = 1;
    }
    size_t target_size = new_size;
    int extra_for_eos = removed_eos ? 1 : 0;
    if ((size_t)tok->encoded_count >= target_size) {
        if (removed_eos) tokenizer_append_eos(tok);
        return;
    }
    int pad = tokenizer_add_vocab(tok, "[PAD]");
    if (tok->pad_id < 0) tok->pad_id = pad;
    if (!tokenizer_reserve_encoded(tok, (int)target_size + extra_for_eos)) {
        if (removed_eos) tokenizer_append_eos(tok);
        return;
    }
    while ((size_t)tok->encoded_count < target_size) {
        tok->encoded[tok->encoded_count++] = pad;
    }
    if (removed_eos) {
        tokenizer_append_eos(tok);
    }
}

static inline const int* tokenizer_data_ptr(const Tokenizer* tok) {
    return tok->encoded;
}

static inline int tokenizer_data_len(const Tokenizer* tok) {
    return tok->encoded_count;
}

static inline int tokenizer_vocab_size(const Tokenizer* tok) {
    return tok->vocab_size;
}

static inline int tokenizer_pad_id(const Tokenizer* tok) {
    return tok->pad_id;
}

static inline int tokenizer_eos_id(const Tokenizer* tok) {
    return tok->eos_id;
}

static inline const char* tokenizer_token_from_id(const Tokenizer* tok, int id) {
    if (tok == NULL || tok->vocab == NULL) return NULL;
    if (id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

static inline void tokenizer_print_tokens(const Tokenizer* tok, const int* token_ids, int len) {
    if (tok == NULL || token_ids == NULL || len <= 0) {
        printf("(no tokens)\n");
        return;
    }
    for (int i = 0; i < len; ++i) {
        const char* token = tokenizer_token_from_id(tok, token_ids[i]);
        if (token == NULL) token = "[UNK]";
        if (i > 0) {
            int add_space = 1;
            size_t token_len = strlen(token);
            if (token_len == 1) {
                unsigned char c = (unsigned char)token[0];
                if (!isalnum(c)) {
                    add_space = 0;
                }
            }
            if (add_space) {
                printf(" ");
            }
        }
        printf("%s", token);
    }
    printf("\n");
}
