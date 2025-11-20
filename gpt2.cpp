#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "embedding.h"
#include "gelu.h"
#include "softmax.h"
#include "linear.h"
#include "layernorm.h"
#include "head.h"
#include "multihead_attention.h"
#include "mlp.h"
#include "block.h"
#include "cross_entropy.h"
#include "gpt.h"

// Helper: fill Linear with constants
static void fill_linear(Linear* layer, float w_val, float b_val) {
    tensor_fill(&layer->weight, w_val);
    if (layer->use_bias) {
        tensor_fill(&layer->bias, b_val);
    }
}

// Helper: fill embeddings
static void fill_embedding(Embedding* emb, float val) {
    tensor_fill(&emb->weight, val);
}

int main() {
    // === GPTConfig ==
    int block_size = 8;
    int vocab_size = 50;
    int n_layer    = 2;
    int n_head     = 2;
    int n_embd     = 16;
    float dropout  = 0.0f;     // we ignore dropout in this C implementation

    // === Initialize GPT ===
    GPT model;
    gpt_init(&model,
             vocab_size,
             block_size,
             n_layer,
             n_head,
             n_embd,
             dropout);

    // === Fill weights with simple constants for debugging ===
    fill_embedding(&model.wte, 0.01f);
    fill_embedding(&model.wpe, 0.02f);

    for (int i = 0; i < model.n_layer; ++i) {
        Block* blk = &model.blocks[i];

        // MHA q/k/v and proj
        for (int h = 0; h < blk->mha.n_heads; ++h) {
            tensor_fill(&blk->mha.heads[h].key.weight,   0.1f);
            tensor_fill(&blk->mha.heads[h].query.weight, 0.1f);
            tensor_fill(&blk->mha.heads[h].value.weight, 0.1f);
        }
        fill_linear(&blk->mha.proj, 0.05f, 0.0f);

        // MLP
        fill_linear(&blk->mlp.c_fc,   0.02f, 0.0f);
        fill_linear(&blk->mlp.c_proj, 0.03f, 0.0f);
    }

    fill_linear(&model.lm_head, 0.01f, 0.0f);

    // === Build input idx ===
    int B = 1;
    int T = 5;
    int idx[5] = {1, 2, 3, 4, 5};

    // === Targets for CE ===
    int targets[5] = {2, 3, 4, 5, 6};

    // === Run forward WITH logits ===
    Tensor logits;
    float loss;

    gpt_forward_with_loss(&model, idx, targets, B, T, &logits, &loss);

    printf("Logits shape: (%d, %d, %d)\n",
           logits.shape[0], logits.shape[1], logits.shape[2]);
    printf("Loss: %f\n\n", loss);

    // === Print some logits ===
    printf("First 5 logits for each position:\n");
    for (int t = 0; t < T; ++t) {
        printf("t=%d: ", t);
        for (int v = 0; v < 5; ++v) {
            float val = tensor_get3(&logits, 0, t, v);
            printf("%f ", val);
        }
        printf("...\n");
    }

    // Cleanup
    tensor_free(&logits);
    gpt_free(&model);

    return 0;
}
