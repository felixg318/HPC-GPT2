#include <stdio.h>
#include <stdlib.h>
#include "gpt.h"
#include "dataloader.h"
#include "adam.h"
#include "autograd.h"

int main() {
    // Hyperparameters
    int vocab_size = 29;
    int block_size = 8;
    int n_layer = 2;
    int n_head = 2;
    int n_embd = 16;
    float dropout_p = 0.1f;
    
    int batch_size = 1;
    int seq_len = 5;
    float lr = 1e-4;
    int epochs = 2;
    float clip_grad_norm_val = 1.0f;

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
    
    adam_init(&optimizer, params, num_params, lr, 0.9f, 0.999f, 1e-8f);
    optimizer.lr_scheduler = linear_lr_decay;
    
    // Initialize dataloader
    DataLoader dl;
    dataloader_init(&dl, "dummy_data.txt", batch_size, seq_len);

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
    
    // Free resources
    gpt_free(&gpt);
    adam_free(&optimizer);
    dataloader_free(&dl);
    free(params);
    
    return 0;
}
