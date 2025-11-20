// test_mha.cpp
#include <stdio.h>
#include "tensor.h"
#include "linear.h"
#include "softmax.h"
#include "head.h"
#include "multihead_attention.h"

// Helper to fill Linear with constants
static void linear_fill_constant(Linear* layer, float w_val, float b_val) {
    tensor_fill(&layer->weight, w_val);
    if (layer->use_bias) {
        tensor_fill(&layer->bias, b_val);
    }
}

int main() {
    int B = 1;
    int T = 3;
    int embed_dim = 4;
    int n_heads = 2;  // head_size = 2

    // 1) Create input x: shape (B, T, C)
    Tensor x;
    int x_shape[3] = {B, T, embed_dim};
    tensor_init(&x, 3, x_shape);

    // Fill x with some simple pattern
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < embed_dim; ++c) {
                float val = (float)(b * 100 + t * 10 + c);
                tensor_set3(&x, b, t, c, val);
            }
        }
    }

    // 2) Init MHA
    MultiHeadAttention mha;
    mha_init(&mha, embed_dim, n_heads, 0.0f /*dropout*/, 1 /*causal*/);

    // 3) For testing: fill q,k,v and proj weights with constants
    for (int h = 0; h < mha.n_heads; ++h) {
        tensor_fill(&mha.heads[h].key.weight,   0.1f);
        tensor_fill(&mha.heads[h].query.weight, 0.2f);
        tensor_fill(&mha.heads[h].value.weight, 0.3f);
        // (all biases are 0 because we used use_bias=0)
    }

    // Final projection: weights=0.5, bias=0.0
    linear_fill_constant(&mha.proj, 0.5f, 0.0f);

    // 4) Run forward
    Tensor out;
    mha_forward(&mha, &x, &out);

    // 5) Print result
    printf("Output shape: (B=%d, T=%d, C=%d)\n",
           out.shape[0], out.shape[1], out.shape[2]);

    for (int b = 0; b < out.shape[0]; ++b) {
        for (int t = 0; t < out.shape[1]; ++t) {
            printf("b=%d, t=%d: ", b, t);
            for (int c = 0; c < out.shape[2]; ++c) {
                float v = tensor_get3(&out, b, t, c);
                printf("%f ", v);
            }
            printf("\n");
        }
    }

    // 6) Cleanup
    tensor_free(&x);
    tensor_free(&out);
    mha_free(&mha);

    return 0;
}
