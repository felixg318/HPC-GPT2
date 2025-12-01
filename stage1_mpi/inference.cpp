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

static void generate_sample_text(GPT* gpt,
                                 const Tokenizer* tokenizer,
                                 const int* seed_tokens,
                                 int seed_len,
                                 int max_new_tokens,
                                 int allow_print) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL) return;
    int total_capacity = seed_len + max_new_tokens;
    int* context = (int*)malloc(total_capacity * sizeof(int));
    if (context == NULL) return;
    for (int i = 0; i < seed_len; ++i) context[i] = seed_tokens[i];
    int current_len = seed_len;
    int eos_id = tokenizer_eos_id(tokenizer);

    while (current_len < total_capacity) {
        int window = current_len < gpt->block_size ? current_len : gpt->block_size;
        const int* window_ptr = context + (current_len - window);
        Tensor logits;
        gpt_forward_logits(gpt, window_ptr, 1, window, &logits);
        int next_token = greedy_select_next_token(&logits);
        tensor_free(&logits);
        context[current_len++] = next_token;
        if (allow_print && eos_id >= 0 && next_token == eos_id) {
            printf("Encountered EOS token at position %d, stopping generation.\n", current_len - 1);
            break;
        }
    }

    if (allow_print) {
        printf("\nGenerated text:\n");
        tokenizer_print_tokens(tokenizer, context, current_len);
        printf("\n");
    }
    free(context);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const char* weights_path = (argc > 1) ? argv[1] : "trained_weights.bin";

    // Hyperparameters must match training
    int block_size = 48;
    int n_layer = 8;
    int n_head = 12;
    int n_embd = 192;
    float dropout_p = 0.1f;
    int batch_size = 8;
    int seq_len = block_size;

    // Build tokenizer on rank 0 and broadcast to others
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, rank == 0 ? "dummy_data.txt" : NULL);
    int tokenizer_ok = 1;
    if (rank == 0) {
        if (!tokenizer_extract(&tokenizer)) tokenizer_ok = 0;
        if (tokenizer_ok && !tokenizer_encode(&tokenizer)) tokenizer_ok = 0;
        if (tokenizer_ok) {
            size_t min_tokens = (size_t)batch_size * seq_len * (size_t)world_size + 1;
            tokenizer_pad_to(&tokenizer, min_tokens);
        }
    }
    MPI_Bcast(&tokenizer_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!tokenizer_ok) {
        if (rank == 0) printf("Failed to build tokenizer.\n");
        tokenizer_free(&tokenizer);
        MPI_Finalize();
        return 1;
    }
    if (!tokenizer_broadcast(&tokenizer, rank)) {
        if (rank == 0) printf("Failed to broadcast tokenizer.\n");
        tokenizer_free(&tokenizer);
        MPI_Finalize();
        return 1;
    }

    int vocab_size = tokenizer_vocab_size(&tokenizer);

    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    gpt_set_distributed(&gpt, rank, world_size);

    TensorPtrArray param_list;
    tensor_ptr_array_init(&param_list);
    gpt_collect_params(&gpt, &param_list);

    int load_ok = 1;
    if (rank == 0) {
        load_ok = load_weights(weights_path, &param_list);
        if (!load_ok) {
            printf("Failed to load weights from %s\n", weights_path);
        }
    }
    MPI_Bcast(&load_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!load_ok) {
        tensor_ptr_array_free(&param_list);
        gpt_free(&gpt);
        tokenizer_free(&tokenizer);
        MPI_Finalize();
        return 1;
    }

    // Broadcast parameters to all ranks
    for (int i = 0; i < param_list.count; ++i) {
        Tensor* param = param_list.data[i];
        int n = tensor_numel(param);
        MPI_Bcast(param->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    auto infer_start = std::chrono::high_resolution_clock::now();

    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);
    if (corpus_len > 0) {
        int prompt_len = seq_len;
        if (prompt_len > corpus_len) prompt_len = corpus_len;
        if (prompt_len > gpt.block_size) prompt_len = gpt.block_size;
        int max_new_tokens = 30;
        generate_sample_text(&gpt, &tokenizer, corpus_tokens, prompt_len, max_new_tokens, rank == 0);
    } else {
        if (rank == 0) {
            printf("Tokenizer has no tokens; nothing to generate.\n");
        }
    }

    auto infer_end = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end - infer_start).count();
    if (rank == 0) {
        printf("Inference time: %.4f seconds (%.2f ms)\n", infer_ms / 1000.0, infer_ms);
    }

    gpt_clear_activations(&gpt);
    gpt_free(&gpt);
    tokenizer_free(&tokenizer);
    tensor_ptr_array_free(&param_list);
    MPI_Finalize();
    return 0;
}
