// inference.cpp
// Load trained weights and run a single text generation pass.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>
#include "gpt.h"
#include "checkpoint.h"
#include "autograd.h"
#include "dataloader.h"

static int greedy_select_next_token(const Tensor* logits) {
    if (logits == NULL || logits->ndim != 3) return 0;
    int V = logits->shape[2];
    int last_t = logits->shape[1] - 1;
    float best_val = tensor_get3(logits, 0, last_t, 0);
    int best_idx = 0;
    for (int v = 1; v < V; ++v) {
        float val = tensor_get3(logits, 0, last_t, v);
        if (val > best_val) {
            best_val = val;
            best_idx = v;
        }
    }
    return best_idx;
}

static int generate_sequence_into_buffer(GPT* gpt,
                                         const Tokenizer* tokenizer,
                                         const int* seed_tokens,
                                         int seed_len,
                                         int max_new_tokens,
                                         int* out_buffer,
                                         int buffer_capacity,
                                         int rank) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL || out_buffer == NULL) {
        if (rank == 0) printf("generate_sequence_into_buffer: invalid inputs\n");
        return -1;
    }
    if (seed_len <= 0) {
        if (rank == 0) printf("generate_sequence_into_buffer: need at least one seed token\n");
        return -1;
    }

    int total_capacity = seed_len + max_new_tokens;
    if (total_capacity > buffer_capacity) {
        if (rank == 0) {
            printf("generate_sequence_into_buffer: buffer too small (%d < %d)\n",
                   buffer_capacity, total_capacity);
        }
        return -1;
    }

    for (int i = 0; i < seed_len; ++i) {
        out_buffer[i] = seed_tokens[i];
    }

    int current_len = seed_len;
    int eos_id = tokenizer_eos_id(tokenizer);

    while (current_len < total_capacity) {
        int window = current_len < gpt->block_size ? current_len : gpt->block_size;
        const int* window_ptr = out_buffer + (current_len - window);

        Tensor logits;
        gpt_forward_logits(gpt, window_ptr, 1, window, &logits);

        int next_token = greedy_select_next_token(&logits);
        tensor_free(&logits);

        out_buffer[current_len++] = next_token;

        if (eos_id >= 0 && next_token == eos_id) {
            break;
        }
    }

    return current_len;
}

static void distributed_generate_text(GPT* gpt,
                                      const Tokenizer* tokenizer,
                                      const int* corpus_tokens,
                                      int corpus_len,
                                      int prompt_len,
                                      int max_new_tokens,
                                      int rank) {
    if (gpt == NULL || tokenizer == NULL || corpus_tokens == NULL) {
        if (rank == 0) printf("distributed_generate_text: missing inputs\n");
        return;
    }
    if (corpus_len <= 0 || prompt_len <= 0) {
        if (rank == 0) {
            printf("Skipping text generation; insufficient tokens (corpus_len=%d, prompt_len=%d).\n",
                   corpus_len, prompt_len);
        }
        return;
    }

    int total_capacity = prompt_len + max_new_tokens;
    if (total_capacity <= 0) {
        if (rank == 0) printf("Skipping text generation; non-positive capacity (%d).\n", total_capacity);
        return;
    }

    int* buffer = (int*)malloc((size_t)total_capacity * sizeof(int));
    if (buffer == NULL) {
        if (rank == 0) printf("distributed_generate_text: failed to allocate buffer\n");
        return;
    }

    const int* prompt_ptr = corpus_tokens;

    auto gen_start = std::chrono::high_resolution_clock::now();
    int produced = -1;
    if (rank == 0) {
        produced = generate_sequence_into_buffer(gpt, tokenizer, prompt_ptr, prompt_len,
                                                 max_new_tokens, buffer, total_capacity, rank);
    }
    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();

    if (rank == 0) {
        if (produced > 0) {
            int new_tokens = produced - prompt_len;
            if (new_tokens < 0) new_tokens = 0;
            printf("\nGenerated sample (%d seed tokens + %d new tokens):\n",
                   prompt_len, new_tokens);
            tokenizer_print_tokens(tokenizer, buffer, produced);
            printf("\nGeneration time: %.6f seconds\n", gen_ms / 1000.0);
        } else {
            printf("Failed to generate text.\n");
        }
    }

    free(buffer);
}

int main(int argc, char** argv) {
    const unsigned int DEFAULT_RANDOM_SEED = 1234u;
    unsigned int seed = DEFAULT_RANDOM_SEED;
    const char* weights_path = (argc > 1) ? argv[1] : "trained_weights.bin";

    int mpi_initialized = 0;
    if (MPI_Init(&argc, &argv) == MPI_SUCCESS) {
        mpi_initialized = 1;
    } else {
        printf("MPI_Init failed\n");
        return 1;
    }

    tensor_set_seed(seed);

    int world_size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Hyperparameters must match training
    int block_size = 48;
    int n_layer = 8;
    int n_head = 12;
    int n_embd = 192;
    float dropout_p = 0.1f;
    int batch_size = 8;
    int seq_len = block_size;

    // Build tokenizer locally
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "dummy_data.txt");
    if (!tokenizer_extract(&tokenizer) || !tokenizer_encode(&tokenizer)) {
        printf("Failed to build tokenizer.\n");
        tokenizer_free(&tokenizer);
        if (mpi_initialized) MPI_Finalize();
        return 1;
    }
    size_t min_tokens = (size_t)batch_size * seq_len * (size_t)world_size + 1;
    tokenizer_pad_to(&tokenizer, min_tokens);

    int vocab_size = tokenizer_vocab_size(&tokenizer);

    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    gpt_set_distributed(&gpt, rank, world_size);

    TensorPtrArray param_list;
    tensor_ptr_array_init(&param_list);
    gpt_collect_params(&gpt, &param_list);

    int load_ok = load_weights(weights_path, &param_list);
    if (!load_ok) {
        printf("Failed to load weights from %s\n", weights_path);
        tensor_ptr_array_free(&param_list);
        gpt_free(&gpt);
        tokenizer_free(&tokenizer);
        if (mpi_initialized) MPI_Finalize();
        return 1;
    }
    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);
    int prompt_len = seq_len;
    if (prompt_len > corpus_len) prompt_len = corpus_len;
    if (prompt_len > gpt.block_size) prompt_len = gpt.block_size;
    int max_new_tokens = 30;
    distributed_generate_text(&gpt, &tokenizer, corpus_tokens, corpus_len,
                              prompt_len, max_new_tokens, rank);

    gpt_clear_activations(&gpt);
    gpt_free(&gpt);
    tokenizer_free(&tokenizer);
    tensor_ptr_array_free(&param_list);
    if (mpi_initialized) MPI_Finalize();
    return 0;
}
