// inference.cpp
// Load trained weights and run a single text generation pass (CUDA, optional MPI).

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "gpt.h"
#include "checkpoint.h"
#include "autograd.h"
#include "dataloader.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

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
                                         int buffer_capacity) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL || out_buffer == NULL) return -1;
    if (seed_len <= 0) return -1;

    int total_capacity = seed_len + max_new_tokens;
    if (total_capacity > buffer_capacity) return -1;

    for (int i = 0; i < seed_len; ++i) out_buffer[i] = seed_tokens[i];

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
        if (eos_id >= 0 && next_token == eos_id) break;
    }

    return current_len;
}

int main(int argc, char** argv) {
    const unsigned int DEFAULT_RANDOM_SEED = 1234u;
    unsigned int seed = DEFAULT_RANDOM_SEED;
    const char* weights_path = (argc > 1) ? argv[1] : "trained_weights.bin";

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif
    tensor_set_seed(seed);

    int world_size = 1;
    int rank = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if (world_size > 1 && rank == 0) {
        printf("Note: stage2 inference is single-sample; only rank 0 will print output.\n");
    }

    // Hyperparameters must match training
    int block_size = 256;
    int n_layer = 6;
    int n_head = 6;
    int n_embd = 384;
    float dropout_p = 0.1f;
    int batch_size = 16;
    int seq_len = block_size;

    // Build tokenizer locally (only rank 0 does work)
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "../data/dummy.txt");
    if (!tokenizer_extract(&tokenizer) || !tokenizer_encode(&tokenizer)) {
        if (rank == 0) printf("Failed to build tokenizer.\n");
        tokenizer_free(&tokenizer);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    size_t min_tokens = (size_t)batch_size * seq_len + 1;
    tokenizer_pad_to(&tokenizer, min_tokens);

    int vocab_size = tokenizer_vocab_size(&tokenizer);

    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    TensorPtrArray param_list;
    tensor_ptr_array_init(&param_list);
    gpt_collect_params(&gpt, &param_list);

    int load_ok = load_weights(weights_path, &param_list);
#ifdef USE_MPI
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (!load_ok) {
        if (rank == 0) printf("Failed to load weights from %s\n", weights_path);
        tensor_ptr_array_free(&param_list);
        gpt_free(&gpt);
        tokenizer_free(&tokenizer);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);
    int prompt_len = seq_len;
    if (prompt_len > corpus_len) prompt_len = corpus_len;
    if (prompt_len > gpt.block_size) prompt_len = gpt.block_size;
    int max_new_tokens = 30;

    int total_capacity = prompt_len + max_new_tokens;
    int* buffer = (int*)malloc((size_t)total_capacity * sizeof(int));
    if (buffer == NULL) {
        if (rank == 0) printf("Failed to allocate output buffer.\n");
        tensor_ptr_array_free(&param_list);
        gpt_free(&gpt);
        tokenizer_free(&tokenizer);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    auto gen_start = std::chrono::high_resolution_clock::now();
    int produced = generate_sequence_into_buffer(&gpt, &tokenizer, corpus_tokens, prompt_len,
                                                 max_new_tokens, buffer, total_capacity);
    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();

    if (rank == 0) {
        if (produced > 0) {
            int new_tokens = produced - prompt_len;
            if (new_tokens < 0) new_tokens = 0;
            printf("\nGenerated sample (%d seed tokens + %d new tokens):\n",
                   prompt_len, new_tokens);
            printf("Seed tokens:\n");
            tokenizer_print_tokens(&tokenizer, buffer, prompt_len);
            printf("New tokens:\n");
            if (new_tokens > 0) {
                tokenizer_print_tokens(&tokenizer, buffer + prompt_len, new_tokens);
            } else {
                printf("(none)\n");
            }
            printf("\n");
            printf("Generation time: %.6f seconds\n", gen_ms / 1000.0);
        } else {
            printf("Failed to generate text.\n");
        }
    }

    free(buffer);
    gpt_clear_activations(&gpt);
    gpt_free(&gpt);
    tokenizer_free(&tokenizer);
    tensor_ptr_array_free(&param_list);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
