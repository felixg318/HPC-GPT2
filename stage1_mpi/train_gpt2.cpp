#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <mpi.h>
#include "gpt.h"
#include "dataloader.h"
#include "adam.h"
#include "autograd.h"
#include "checkpoint.h"

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
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL) {
        if (allow_print) {
            printf("generate_sample_text: missing inputs\n");
        }
        return;
    }
    if (seed_len <= 0) {
        if (allow_print) {
            printf("generate_sample_text: need at least one seed token\n");
        }
        return;
    }

    int total_capacity = seed_len + max_new_tokens;
    int* context = (int*)malloc(total_capacity * sizeof(int));
    if (context == NULL) {
        if (allow_print) {
            printf("generate_sample_text: failed to allocate context buffer\n");
        }
        return;
    }
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

    int generated_tokens = current_len - seed_len;
    if (allow_print) {
        printf("\nGenerated sample (%d seed tokens + %d new tokens):\n", seed_len, generated_tokens);
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

    // Hyperparameters 
    int block_size = 48;   // n_ctx / n_positions
    int n_layer = 8;
    int n_head = 12;  // ensure num_ranks <= n_head or expect some ranks to sit idle.
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
            printf("batch_size (%d) must be divisible by world_size (%d) for consistent data-parallel tokens.\n",
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
            size_t min_tokens = (size_t)batch_size * seq_len * (size_t)world_size + 1;
            tokenizer_pad_to(&tokenizer, min_tokens);
        }
    }
    MPI_Bcast(&tokenizer_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!tokenizer_ok) {
        if (rank == 0) tokenizer_free(&tokenizer);
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

    MPI_Barrier(MPI_COMM_WORLD);
    auto train_start = std::chrono::high_resolution_clock::now();
    long long tokens_per_epoch = (long long)batch_size * (long long)seq_len * (long long)world_size;
    long long total_tokens = tokens_per_epoch * (long long)epochs;
    if (rank == 0) {
        printf("Training tokens (MPI): per epoch=%lld, total=%lld (world_size=%d)\n",
               tokens_per_epoch, total_tokens, world_size);
    }
    const float inv_world = (world_size > 0) ? (1.0f / (float)world_size) : 1.0f;

    // Precompute flattened grad buffer for a single Allreduce per step.
    int total_grad_elems = 0;
    for (int i = 0; i < param_list.count; ++i) {
        total_grad_elems += tensor_numel(param_list.data[i]);
    }
    float* grad_buffer = (float*)malloc((size_t)total_grad_elems * sizeof(float));
    if (grad_buffer == NULL) {
        if (rank == 0) printf("Failed to allocate grad_buffer\n");
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
        int* inputs;
        int* targets;
        dataloader_next_batch(&dl, &inputs, &targets);
        
        // Forward pass
        Tensor logits, loss;
        gpt_forward_with_loss(&gpt, inputs, targets, batch_size, seq_len, &logits, &loss);
        float local_loss = loss.data[0];
        float avg_loss = 0.0f;
        MPI_Allreduce(&local_loss, &avg_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        avg_loss *= inv_world;
        
        // Backward pass
        backward(&loss);
        gpt_clear_activations(&gpt);

        if (world_size > 1) {
            // Pack grads into contiguous buffer for one Allreduce.
            int offset = 0;
            for (int i = 0; i < param_list.count; ++i) {
                Tensor* param = param_list.data[i];
                int n = tensor_numel(param);
                if (n <= 0) continue;
                for (int j = 0; j < n; ++j) {
                    grad_buffer[offset + j] = param->grad[j];
                }
                offset += n;
            }
            MPI_Allreduce(MPI_IN_PLACE, grad_buffer, total_grad_elems, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            // Unpack averaged grads back into params.
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
    if (rank == 0) {
        if (save_weights("trained_weights.bin", &param_list)) {
            printf("Saved trained weights to trained_weights.bin\n");
        } else {
            printf("Failed to save trained weights.\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
    if (rank == 0) {
        printf("Total training time: %.8f seconds (%.4f minutes)\n", train_ms / 1000.0, train_ms / 60000.0);
    }

    // Parallel text generation (each rank handles a slice, only rank 0 prints)
    MPI_Barrier(MPI_COMM_WORLD);
    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);
    double gen_ms_local = 0.0;
    if (corpus_len > 0) {
        int prompt_len = seq_len;
        if (prompt_len > corpus_len) prompt_len = corpus_len;
        if (prompt_len > gpt.block_size) prompt_len = gpt.block_size;
        int max_new_tokens = 30;

        int start = (rank * prompt_len) % (corpus_len - prompt_len + 1);
        const int* prompt_ptr = corpus_tokens + start;

        auto gen_start = std::chrono::high_resolution_clock::now();
        generate_sample_text(&gpt, &tokenizer, prompt_ptr, prompt_len, max_new_tokens, rank == 0);
        auto gen_end = std::chrono::high_resolution_clock::now();
        gen_ms_local = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
    }
    double gen_ms_max = 0.0;
    MPI_Reduce(&gen_ms_local, &gen_ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (corpus_len > 0) {
            printf("Text generation time: %.8f seconds\n", gen_ms_max / 1000.0);
        } else {
            printf("Skipping text generation; tokenizer has no tokens.\n");
        }
    }
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
