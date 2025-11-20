// test_block.cpp
#include <stdio.h>
#include "tensor.h"
#include "softmax.h"
#include "gelu.h"
#include "linear.h"
#include "layernorm.h"
#include "head.h"
#include "multihead_attention.h"
#include "mlp.h"
#include "block.h"

int main() {
    int B = 1;
    int T = 3;
    int C = 4;      // embed_dim
    int n_heads = 2;

    // 1) Input x: (B,T,C)
    Tensor x;
    int x_shape[3] = {B, T, C};
    tensor_init(&x, 3, x_shape);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                float val = (float)(b * 100 + t * 10 + c);
                tensor_set3(&x, b, t, c, val);
            }
        }
    }

    // 2) Init block
    Block blk;
    block_init(&blk, C, n_heads, 0.0f /*attn dropout*/, 0.0f /*mlp dropout*/, 1 /*causal*/);

    // 3) For testing, fill attention weights with constants
    for (int h = 0; h < blk.mha.n_heads; ++h) {
        tensor_fill(&blk.mha.heads[h].key.weight,   0.1f);
        tensor_fill(&blk.mha.heads[h].query.weight, 0.2f);
        tensor_fill(&blk.mha.heads[h].value.weight, 0.3f);
    }
    // Final projection of MHA
    linear_fill_constant(&blk.mha.proj, 0.5f, 0.0f);

    // MLP is already initialized in mlp_init with some constants (see mlp.h).
    // If you want, you can override:
    // linear_fill_constant(&blk.mlp.c_fc,   0.1f, 0.01f);
    // linear_fill_constant(&blk.mlp.c_proj, 0.05f, 0.0f);

    // 4) Forward through the block
    Tensor y;
    block_forward(&blk, &x, &y);

    // 5) Print results
    printf("Block output shape: (B=%d, T=%d, C=%d)\n",
           y.shape[0], y.shape[1], y.shape[2]);

    for (int b = 0; b < y.shape[0]; ++b) {
        for (int t = 0; t < y.shape[1]; ++t) {
            printf("b=%d, t=%d: ", b, t);
            for (int c = 0; c < y.shape[2]; ++c) {
                float v = tensor_get3(&y, b, t, c);
                printf("%f ", v);
            }
            printf("\n");
        }
    }

    // 6) Cleanup
    tensor_free(&x);
    tensor_free(&y);
    block_free(&blk);

    return 0;
}
