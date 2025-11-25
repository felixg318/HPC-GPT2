#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "gpt.h"
#include "dataloader.h"
#include "adam.h"
#include "autograd.h"

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
                                 int max_new_tokens) {
    if (gpt == NULL || tokenizer == NULL || seed_tokens == NULL) {
        printf("generate_sample_text: missing inputs\n");
        return;
    }
    if (seed_len <= 0) {
        printf("generate_sample_text: need at least one seed token\n");
        return;
    }

    int total_capacity = seed_len + max_new_tokens;
    int* context = (int*)malloc(total_capacity * sizeof(int));
    if (context == NULL) {
        printf("generate_sample_text: failed to allocate context buffer\n");
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
        if (eos_id >= 0 && next_token == eos_id) {
            printf("Encountered EOS token at position %d, stopping generation.\n", current_len - 1);
            break;
        }
    }

    int generated_tokens = current_len - seed_len;
    printf("\nGenerated sample (%d seed tokens + %d new tokens):\n", seed_len, generated_tokens);
    tokenizer_print_tokens(tokenizer, context, current_len);
    printf("\n");
    free(context);
}

int main() {
    // Tokenize training corpus
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "dummy_data.txt");
    if (!tokenizer_extract(&tokenizer)) {
        tokenizer_free(&tokenizer);
        return 1;
    }
    if (!tokenizer_encode(&tokenizer)) {
        tokenizer_free(&tokenizer);
        return 1;
    }
    
    // Hyperparameters (aligned with the GPT-2 config)
    int block_size = 1024;   // n_ctx / n_positions
    int n_layer = 12;
    int n_head = 12;
    int n_embd = 768;
    float dropout_p = 0.1f;  // resid/embd/attn dropout
    
    int batch_size = 2;
    int seq_len = block_size;
    float lr = 1e-3f;
    int epochs = 1;
    float clip_grad_norm_val = 1.0f;

    size_t min_tokens = (size_t)batch_size * seq_len + 1;
    tokenizer_pad_to(&tokenizer, min_tokens);
    int vocab_size = tokenizer_vocab_size(&tokenizer);
    
    // Initialize model
    GPT gpt;
    gpt_init(&gpt, vocab_size, block_size, n_layer, n_head, n_embd, dropout_p);
    
    // Initialize optimizer
    AdamOptimizer optimizer;
    // We need to collect all parameters first.
    // This is a simplification. A real implementation would have a more robust way to get all parameters.
    int num_params = 0;
    Tensor** params = (Tensor**)malloc(1000 * sizeof(Tensor*)); // Assuming max 1000 parameter tensors
    
    params[num_params++] = &gpt.wte.weight;
    params[num_params++] = &gpt.wpe.weight;
    for (int i = 0; i < n_layer; ++i) {
        params[num_params++] = &gpt.blocks[i].ln1.gamma;
        params[num_params++] = &gpt.blocks[i].ln1.beta;
        params[num_params++] = &gpt.blocks[i].ln2.gamma;
        params[num_params++] = &gpt.blocks[i].ln2.beta;
        for (int h = 0; h < n_head; ++h) {
            params[num_params++] = &gpt.blocks[i].mha.heads[h].query.weight;
            params[num_params++] = &gpt.blocks[i].mha.heads[h].key.weight;
            params[num_params++] = &gpt.blocks[i].mha.heads[h].value.weight;
        }
        params[num_params++] = &gpt.blocks[i].mha.proj.weight;
        params[num_params++] = &gpt.blocks[i].mha.proj.bias;
        params[num_params++] = &gpt.blocks[i].mlp.c_fc.weight;
        params[num_params++] = &gpt.blocks[i].mlp.c_fc.bias;
        params[num_params++] = &gpt.blocks[i].mlp.c_proj.weight;
        params[num_params++] = &gpt.blocks[i].mlp.c_proj.bias;
    }
    params[num_params++] = &gpt.ln_f.gamma;
    params[num_params++] = &gpt.ln_f.beta;
    params[num_params++] = &gpt.lm_head.weight;
    
    adam_init(&optimizer, params, num_params, lr, 0.9f, 0.95f, 1e-8f);
    optimizer.lr_scheduler = linear_lr_decay;
    
    // Initialize dataloader
    DataLoader dl;
    dataloader_init_with_tokenizer(&dl, &tokenizer, batch_size, seq_len);

    auto train_start = std::chrono::high_resolution_clock::now();

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int* inputs;
        int* targets;
        dataloader_next_batch(&dl, &inputs, &targets);
        
        // Forward pass
        Tensor logits, loss;
        gpt_forward_with_loss(&gpt, inputs, targets, batch_size, seq_len, &logits, &loss);
        
        // Backward pass
        backward(&loss);
        gpt_clear_activations(&gpt);
        
        // Update weights
        adam_step(&optimizer, clip_grad_norm_val);
        
        // Zero gradients
        adam_zero_grad(&optimizer);
        
        printf("Epoch %d, Loss: %f\n", epoch, loss.data[0]);
        
        free(inputs);
        free(targets);
        tensor_free(&logits);
        tensor_free(&loss);
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
    printf("Total training time: %.2f seconds (%.2f minutes)\n", train_ms / 1000.0, train_ms / 60000.0);

    // Simple text generation demo to inspect model behavior post-training
    const int* corpus_tokens = tokenizer_data_ptr(&tokenizer);
    int corpus_len = tokenizer_data_len(&tokenizer);
    if (corpus_len > 0) {
        int prompt_len = seq_len;
        if (prompt_len > corpus_len) prompt_len = corpus_len;
        if (prompt_len > gpt.block_size) prompt_len = gpt.block_size;
        int max_new_tokens = 30;
        generate_sample_text(&gpt, &tokenizer, corpus_tokens, prompt_len, max_new_tokens);
    } else {
        printf("Skipping text generation; tokenizer has no tokens.\n");
    }
    
    // Free resources
    gpt_clear_activations(&gpt);
    gpt_free(&gpt);
    adam_free(&optimizer);
    dataloader_free(&dl);
    dataloader_free(&dl);
    tokenizer_free(&tokenizer);
    free(params);
    
    return 0;
}
