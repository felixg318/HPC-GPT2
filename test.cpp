#include <stdio.h>
#include "gpt.h"
#include "tensor.h"

// Simple helpers
static void fill_linear(Linear* layer, float w_val, float b_val) {
    tensor_fill(&layer->weight, w_val);
    if (layer->use_bias) tensor_fill(&layer->bias, b_val);
}
static void fill_embedding(Embedding* emb, float val) {
    tensor_fill(&emb->weight, val);
}

int main() {
    int vocab_size = 20;
    int block_size = 8;
    int n_layer    = 1;
    int n_head     = 2;
    int n_embd     = 4;
    float dropout  = 0.0f;

    GPT g;
    gpt_init(&g, vocab_size, block_size, n_layer, n_head, n_embd, dropout);

    // Simple constant init
    fill_embedding(&g.wte, 0.01f);
    fill_embedding(&g.wpe, 0.02f);
    for (int i = 0; i < g.n_layer; ++i) {
        Block* blk = &g.blocks[i];
        for (int h = 0; h < blk->mha.n_heads; ++h) {
            tensor_fill(&blk->mha.heads[h].key.weight,   0.1f);
            tensor_fill(&blk->mha.heads[h].query.weight, 0.1f);
            tensor_fill(&blk->mha.heads[h].value.weight, 0.1f);
        }
        fill_linear(&blk->mha.proj, 0.1f, 0.0f);
        fill_linear(&blk->mlp.c_fc,   0.05f, 0.01f);
        fill_linear(&blk->mlp.c_proj, 0.04f, 0.0f);
    }
    fill_linear(&g.lm_head, 0.03f, 0.0f);

    int B = 1;
    int T = 3;
    int idx[3]     = {1, 2, 3};
    int targets[3] = {2, 3, 4};

    Tensor logits;
    float loss;

    gpt_forward_with_loss(&g, idx, targets, B, T, &logits, &loss);

    printf("Logits shape: (%d,%d,%d)\n",
           logits.shape[0], logits.shape[1], logits.shape[2]);
    printf("Loss: %f\n", loss);

    tensor_free(&logits);
    gpt_free(&g);
    return 0;
}
