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

static int generate_sequence_into_buffer(GPT* gpt,
                                         const Tokenizer* tokenizer,
                                         const int* seed_tokens,
                                         int seed_len,
                                         int max_new_tokens,
                                         int* out_buffer,
                                         int buffer_capacity,
                                         int rank) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL || out_buffer == NULL) {
        printf("[Rank %d] generate_sequence_into_buffer: invalid inputs\n", rank);
        return -1;
    }
    if (seed_len <= 0) {
        printf("[Rank %d] generate_sequence_into_buffer: need at least one seed token\n", rank);
        return -1;
    }

    int total_capacity = seed_len + max_new_tokens;
    if (total_capacity > buffer_capacity) {
        printf("[Rank %d] generate_sequence_into_buffer: buffer too small (%d < %d)\n",
               rank, buffer_capacity, total_capacity);
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
                                      int rank,
                                      int world_size) {
    if (gpt == NULL || tokenizer == NULL || corpus_tokens == NULL) {
        if (rank == 0) {
            printf("distributed_generate_text: missing inputs\n");
        }
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
        if (rank == 0) {
            printf("Skipping text generation; non-positive capacity (%d).\n", total_capacity);
        }
        return;
    }

    int* local_buffer = (int*)malloc((size_t)total_capacity * sizeof(int));
    if (local_buffer == NULL) {
        printf("[Rank %d] distributed_generate_text: failed to allocate buffer\n", rank);
        return;
    }

    int prompt_offset = 0;
    if (corpus_len > prompt_len) {
        int span = corpus_len - prompt_len + 1;
        long long stride = prompt_len > 0 ? prompt_len : 1;
        prompt_offset = (int)(((long long)rank * stride) % (long long)span);
    }

    const int* prompt_ptr = corpus_tokens + prompt_offset;

    MPI_Barrier(MPI_COMM_WORLD);
    auto local_start = std::chrono::high_resolution_clock::now();
    int produced = generate_sequence_into_buffer(gpt, tokenizer, prompt_ptr, prompt_len,
                                                 max_new_tokens, local_buffer, total_capacity,
                                                 rank);
    auto local_end = std::chrono::high_resolution_clock::now();
    double local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(local_end - local_start).count();

    if (produced < 0) {
        produced = 0;
    }
    for (int i = produced; i < total_capacity; ++i) {
        local_buffer[i] = -1;
    }

    int* all_lengths = NULL;
    int* all_offsets = NULL;
    double* all_times = NULL;
    if (rank == 0) {
        all_lengths = (int*)malloc(world_size * sizeof(int));
        all_offsets = (int*)malloc(world_size * sizeof(int));
        all_times = (double*)malloc(world_size * sizeof(double));
    }

    MPI_Gather(&produced, 1, MPI_INT, all_lengths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&prompt_offset, 1, MPI_INT, all_offsets, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_ms, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* gathered_tokens = NULL;
    if (rank == 0) {
        gathered_tokens = (int*)malloc((size_t)world_size * total_capacity * sizeof(int));
    }

    MPI_Gather(local_buffer, total_capacity, MPI_INT,
               gathered_tokens, total_capacity, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nDistributed text generation (%d ranks): prompt_len=%d, max_new_tokens=%d\n",
               world_size, prompt_len, max_new_tokens);
        for (int r = 0; r < world_size; ++r) {
            printf("[Rank %d] prompt_offset=%d, tokens=%d, time=%.4f s\n",
                   r,
                   all_offsets ? all_offsets[r] : 0,
                   all_lengths ? all_lengths[r] : 0,
                   all_times ? (all_times[r] / 1000.0) : 0.0);
            int len = all_lengths ? all_lengths[r] : 0;
            if (len > 0 && gathered_tokens != NULL) {
                tokenizer_print_tokens(tokenizer,
                                       gathered_tokens + r * total_capacity,
                                       len);
                printf("\n");
            } else {
                printf("(no tokens generated)\n");
            }
        }
    }

    if (rank == 0) {
        free(all_lengths);
        free(all_offsets);
        free(all_times);
        free(gathered_tokens);
    }

    free(local_buffer);
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

    // Hyperparameters
    int block_size = 48;   // n_ctx / n_positions
    int n_layer = 8;
    int n_head = 12;       // ensure num_ranks <= n_head or expect some ranks to sit idle.
    int n_embd = 192;
    float dropout_p = 0.1f;  // resid/embd/attn dropout which is not used at all.

    int global_batch_size = 4;
    int seq_len = block_size;
    float lr = 3e-4f;
    int epochs = 8;
    float clip_grad_norm_val = 1.0f;

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

    int batch_size = global_batch_size / world_size;
    if (batch_size <= 0) {
        if (rank == 0) {
            printf("Per-rank batch_size became zero. Increase global_batch_size.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Tokenize training corpus
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, rank == 0 ? "dummy_data.txt" : NULL);

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
    if (!tokenizer_ok) {
        if (rank == 0) {
            tokenizer_free(&tokenizer);
        }
        MPI_Finalize();
        return 1;
    }

    if (!tokenizer_broadcast(&tokenizer, rank)) {
        tokenizer_free(&tokenizer);
        MPI_Finalize();
        return 1;
    }

    int vocab_size = tokenizer_vocab_size(&tokenizer);

    // Initialize model
    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    gpt_set_distributed(&gpt, rank, world_size);

    TensorPtrArray param_list;
    tensor_ptr_array_init(&param_list);
    gpt_collect_params(&gpt, &param_list);

    // Ensure all ranks start from the same random initialization (rank 0 is source)
    if (world_size > 1) {
        for (int i = 0; i < param_list.count; ++i) {
            Tensor* param = param_list.data[i];
            int n = tensor_numel(param);
            MPI_Bcast(param->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }

    AdamOptimizer optimizer;
    adam_init(&optimizer, param_list.data, param_list.count, lr, 0.9f, 0.95f, 1e-8f);
    optimizer.lr_scheduler = linear_lr_decay;
    adam_set_distributed(&optimizer, rank, world_size);

    // Initialize dataloader
    DataLoader dl;
    dataloader_init_with_tokenizer(&dl, &tokenizer, batch_size, seq_len);
    dataloader_set_distributed(&dl, rank, world_size);
    if (!dataloader_broadcast(&dl, 0)) {
        dataloader_free(&dl);
        tokenizer_free(&tokenizer);
        gpt_free(&gpt);
        adam_free(&optimizer);
        tensor_ptr_array_free(&param_list);
        MPI_Finalize();
        return 1;
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

    int total_grad_elems = 0;
    for (int i = 0; i < param_list.count; ++i) {
        total_grad_elems += tensor_numel(param_list.data[i]);
    }

    float* grad_buffer = (float*)malloc((size_t)total_grad_elems * sizeof(float));
    if (grad_buffer == NULL) {
        if (rank == 0) {
            printf("Failed to allocate grad_buffer\n");
        }
        dataloader_free(&dl);
        tokenizer_free(&tokenizer);
        gpt_free(&gpt);
        adam_free(&optimizer);
        tensor_ptr_array_free(&param_list);
        MPI_Finalize();
        return 1;
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

        int offset = 0;
        for (int i = 0; i < param_list.count; ++i) {
            Tensor* param = param_list.data[i];
            int n = tensor_numel(param);
            if (n <= 0) continue;
            memcpy(grad_buffer + offset, param->grad, n * sizeof(float));
            offset += n;
        }

        MPI_Allreduce(MPI_IN_PLACE, grad_buffer, total_grad_elems, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        offset = 0;
        for (int i = 0; i < param_list.count; ++i) {
            Tensor* param = param_list.data[i];
            int n = tensor_numel(param);
            if (n <= 0) continue;
            for (int j = 0; j < n; ++j) {
                param->grad[j] = grad_buffer[offset + j] * inv_world;
            }
            offset += n;
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

    MPI_Finalize();
    return 0;
}
