#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memcpy and seed parsing
#include <chrono>
#include <mpi.h>

#include "gpt.h"
#include "dataloader.h"
#include "adam.h"
#include "autograd.h"
#include "checkpoint.h"

typedef struct {
    Tensor* tensor;
    int offset;
    int length;
    int is_shared;
} ParamSyncInfo;

static inline int all_ranks_ok(int local_ok) {
    int global_ok = local_ok;
    MPI_Allreduce(MPI_IN_PLACE, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global_ok;
}

static inline void abort_all(const char* msg, int rank) {
    if (rank == 0 && msg != NULL) {
        printf("%s\n", msg);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static unsigned int parse_seed_arg(int argc, char** argv, unsigned int default_seed) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            return (unsigned int)atoi(argv[++i]);
        }
        if (strncmp(argv[i], "--seed=", 7) == 0) {
            return (unsigned int)atoi(argv[i] + 7);
        }
    }
    return default_seed;
}

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

static inline void synchronize_logits_across_ranks(Tensor* logits, int world_size) {
    if (logits == NULL || logits->data == NULL || world_size <= 1) return;
    int n = tensor_numel(logits);
    if (n <= 0) return;
    MPI_Allreduce(MPI_IN_PLACE, logits->data, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float inv = 1.0f / (float)world_size;
    for (int i = 0; i < n; ++i) {
        logits->data[i] *= inv;
    }
}

static int generate_sequence_into_buffer(GPT* gpt,
                                         const Tokenizer* tokenizer,
                                         const int* seed_tokens,
                                         int seed_len,
                                         int max_new_tokens,
                                         int* out_buffer,
                                         int buffer_capacity,
                                         int rank,
                                         int world_size) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL || out_buffer == NULL) {
        if (rank == 0) {
            printf("generate_sequence_into_buffer: invalid inputs\n");
        }
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
        synchronize_logits_across_ranks(&logits, world_size);

        int next_token = 0;
        if (rank == 0) {
            next_token = greedy_select_next_token(&logits);
        }
        MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);
        tensor_free(&logits);

        out_buffer[current_len++] = next_token;

        if (eos_id >= 0 && next_token == eos_id) {
            if (rank == 0) {
                printf("Encountered EOS token at position %d, stopping generation.\n", current_len - 1);
            }
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
                                      int rank,
                                      int world_size) {
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto gen_start = std::chrono::high_resolution_clock::now();
    int produced = generate_sequence_into_buffer(gpt, tokenizer, prompt_ptr, prompt_len,
                                                 max_new_tokens, buffer, total_capacity, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
    double max_ms = gen_ms;
    double avg_ms = gen_ms;
    MPI_Allreduce(MPI_IN_PLACE, &max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &avg_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_ms /= (double)world_size;

    if (rank == 0) {
        if (produced > 0) {
            int new_tokens = produced - prompt_len;
            if (new_tokens < 0) new_tokens = 0;
            printf("\nGenerated sample (%d seed tokens + %d new tokens):\n",
                   prompt_len, new_tokens);
            tokenizer_print_tokens(tokenizer, buffer, produced);
            printf("\n");
            printf("Text generation time: %.8f seconds\n", max_ms / 1000.0);
        } else {
            printf("Failed to generate text.\n");
        }
    }

    free(buffer);
}

int main(int argc, char** argv) {
    const unsigned int DEFAULT_RANDOM_SEED = 1234u;
    unsigned int seed = parse_seed_arg(argc, argv, DEFAULT_RANDOM_SEED);

    MPI_Init(&argc, &argv);
    tensor_set_seed(seed);

    int world_size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Hyperparameters (aligned with the GPT-2 config)
    int block_size = 256;
    int n_layer = 6;
    int n_head = 6;
    int n_embd = 384;
    float dropout_p = 0.1f;  // resid/embd/attn dropout which is not used at all.
    
    int batch_size = 16;
    int seq_len = block_size;
    float lr = 3e-4f;
    int epochs = 50;
    float clip_grad_norm_val = 1.0f;

    int global_batch_size = batch_size;

    if (world_size <= 0) {
        if (rank == 0) {
            printf("Invalid world size %d\n", world_size);
        }
        MPI_Finalize();
        return 1;
    }

    if (global_batch_size % world_size != 0) {
        if (rank == 0) {
            printf("batch_size (%d) must be divisible by world_size (%d) for consistent gradients.\n",
                   global_batch_size, world_size);
        }
        MPI_Finalize();
        return 1;
    }

    batch_size = global_batch_size / world_size;
    if (batch_size <= 0) {
        if (rank == 0) {
            printf("Per-rank batch_size became zero. Increase global_batch_size.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Tokenize training corpus
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, rank == 0 ? "../data/tinyshakespeare.txt" : NULL);

    int tokenizer_ok = 1;
    if (rank == 0) {
        if (!tokenizer_extract(&tokenizer)) {
            tokenizer_ok = 0;
        }
        if (tokenizer_ok && !tokenizer_encode(&tokenizer)) {
            tokenizer_ok = 0;
        }
        if (tokenizer_ok) {
            size_t min_tokens = (size_t)global_batch_size * seq_len + 1;
            tokenizer_pad_to(&tokenizer, min_tokens);
        }
    }

    MPI_Bcast(&tokenizer_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!all_ranks_ok(tokenizer_ok)) {
        tokenizer_free(&tokenizer);
        abort_all("Failed to build tokenizer", rank);
    }

    int tok_bcast_ok = tokenizer_broadcast(&tokenizer, rank);
    if (!all_ranks_ok(tok_bcast_ok)) {
        tokenizer_free(&tokenizer);
        abort_all("Failed to broadcast tokenizer", rank);
    }

    int vocab_size = tokenizer_vocab_size(&tokenizer);

    // Initialize model
    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    gpt_set_distributed(&gpt, rank, world_size);

    TensorPtrArray param_list;
    tensor_ptr_array_init(&param_list);
    gpt_collect_params(&gpt, &param_list);

    ParamSyncInfo* sync_info = (ParamSyncInfo*)malloc((size_t)param_list.count * sizeof(ParamSyncInfo));
    int* shared_mask = (int*)malloc((size_t)param_list.count * sizeof(int));
    int sync_alloc_ok = (sync_info != NULL && shared_mask != NULL) ? 1 : 0;
    if (!all_ranks_ok(sync_alloc_ok)) {
        if (sync_info != NULL) free(sync_info);
        if (shared_mask != NULL) free(shared_mask);
        tokenizer_free(&tokenizer);
        gpt_free(&gpt);
        tensor_ptr_array_free(&param_list);
        abort_all("Failed to allocate parameter sync metadata", rank);
    }

    int total_grad_elems = 0;
    for (int i = 0; i < param_list.count; ++i) {
        Tensor* param = param_list.data[i];
        int n = tensor_numel(param);
        sync_info[i].tensor = param;
        sync_info[i].offset = total_grad_elems;
        sync_info[i].length = n;
        int shared = tensor_is_replicated(param);
        sync_info[i].is_shared = shared;
        shared_mask[i] = shared;
        total_grad_elems += n;
    }

    // Ensure all ranks start from the same random initialization (rank 0 is source)
    if (world_size > 1) {
        for (int i = 0; i < param_list.count; ++i) {
            int n = sync_info[i].length;
            if (n <= 0) continue;
            if (!sync_info[i].is_shared) continue;
            MPI_Bcast(param_list.data[i]->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }

    AdamOptimizer optimizer;
    adam_init(&optimizer, param_list.data, param_list.count, lr, 0.9f, 0.95f, 1e-8f);
    optimizer.lr_scheduler = linear_lr_decay;
    adam_set_distributed(&optimizer, rank, world_size);
    adam_set_shared_mask(&optimizer, shared_mask);
    free(shared_mask);

    // Initialize dataloader
    DataLoader dl;
    dataloader_init_with_tokenizer(&dl, &tokenizer, batch_size, seq_len);
    dataloader_set_distributed(&dl, rank, world_size);
    int dl_ok = dataloader_broadcast(&dl, 0);
    if (!all_ranks_ok(dl_ok)) {
        dataloader_free(&dl);
        tokenizer_free(&tokenizer);
        gpt_free(&gpt);
        adam_free(&optimizer);
        tensor_ptr_array_free(&param_list);
        free(sync_info);
        abort_all("Failed to broadcast dataloader", rank);
    }
    dataloader_set_distributed(&dl, rank, world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto train_start = std::chrono::high_resolution_clock::now();

    long long tokens_per_epoch = (long long)global_batch_size * (long long)seq_len;
    long long total_tokens = tokens_per_epoch * (long long)epochs;
    if (rank == 0) {
        printf("Training tokens (MPI): per epoch=%lld, total=%lld (world_size=%d)\n",
               tokens_per_epoch, total_tokens, world_size);
    }

    const float inv_world = (world_size > 0) ? (1.0f / (float)world_size) : 1.0f;

    float* grad_buffer = (float*)malloc((size_t)total_grad_elems * sizeof(float));
    if (!all_ranks_ok(grad_buffer != NULL ? 1 : 0)) {
        dataloader_free(&dl);
        tokenizer_free(&tokenizer);
        gpt_free(&gpt);
        adam_free(&optimizer);
        tensor_ptr_array_free(&param_list);
        free(sync_info);
        if (grad_buffer != NULL) free(grad_buffer);
        abort_all("Failed to allocate grad_buffer", rank);
    }

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int* inputs = NULL;
        int* targets = NULL;
        dataloader_next_batch(&dl, &inputs, &targets);

        // Forward pass
        Tensor logits, loss;
        gpt_forward_with_loss(&gpt, inputs, targets, batch_size, seq_len, &logits, &loss);

        float local_loss = loss.data[0];
        float avg_loss = local_loss;
        MPI_Allreduce(&local_loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        avg_loss *= inv_world;

        // Backward pass
        backward(&loss);
        gpt_clear_activations(&gpt);

        for (int i = 0; i < param_list.count; ++i) {
            int n = sync_info[i].length;
            if (n <= 0) continue;
            memcpy(grad_buffer + sync_info[i].offset,
                   sync_info[i].tensor->grad,
                   n * sizeof(float));
        }

        if (total_grad_elems > 0) {
            MPI_Allreduce(MPI_IN_PLACE, grad_buffer, total_grad_elems, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        for (int i = 0; i < param_list.count; ++i) {
            int n = sync_info[i].length;
            if (n <= 0) continue;
            float scale = sync_info[i].is_shared ? inv_world : 1.0f;
            float* dst = sync_info[i].tensor->grad;
            float* src = grad_buffer + sync_info[i].offset;
            if (scale == 1.0f) {
                memcpy(dst, src, n * sizeof(float));
            } else {
                for (int j = 0; j < n; ++j) {
                    dst[j] = src[j] * scale;
                }
            }
        }

        // Update weights
        adam_step(&optimizer, clip_grad_norm_val);

        // Zero gradients
        adam_zero_grad(&optimizer);

        if (rank == 0) {
            printf("Epoch %d, Loss: %f\n", epoch, avg_loss);
        }

        free(inputs);
        free(targets);
        tensor_free(&logits);
        tensor_free(&loss);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();

    if (rank == 0) {
        printf("Total training time: %.8f seconds (%.4f minutes)\n",
               train_ms / 1000.0, train_ms / 60000.0);
    }

    // Save weights from rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (save_weights("trained_weights.bin", &param_list)) {
            printf("Saved trained weights to trained_weights.bin\n");
        } else {
            printf("Failed to save trained weights.\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);

    MPI_Barrier(MPI_COMM_WORLD);
    int prompt_len = seq_len;
    if (prompt_len > corpus_len) {
        prompt_len = corpus_len;
    }
    if (prompt_len > gpt.block_size) {
        prompt_len = gpt.block_size;
    }
    int max_new_tokens = 30;
    distributed_generate_text(&gpt, &tokenizer, corpus_tokens, corpus_len,
                              prompt_len, max_new_tokens, rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Free resources
    gpt_clear_activations(&gpt);
    gpt_free(&gpt);
    adam_free(&optimizer);
    dataloader_free(&dl);
    tokenizer_free(&tokenizer);
    tensor_ptr_array_free(&param_list);
    free(grad_buffer);
    free(sync_info);

    MPI_Finalize();
    return 0;
}
