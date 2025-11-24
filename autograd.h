// autograd.h
// Autograd engine.

#pragma once

#include "tensor.h"
#include "add.h"
#include "softmax.h"
#include "linear.h"
#include "layernorm.h"
#include "matmul.h"
#include "transpose.h"
#include "head.h"
#include "multihead_attention.h"
#include "gelu.h"
#include "mlp.h"
#include "cross_entropy.h"
#include "embedding.h"


// Build a topological sort of the computation graph.
static inline void build_topo(Tensor* v, Tensor** topo, int* n_topo) {
    if (v->visited) {
        return;
    }
    v->visited = 1;

    // This is a simplification. A real implementation would have a standardized way
    // to get the inputs from the context.
    if (v->_ctx != NULL) {
        if (v->_backward == add_backward) {
            AddContext* ctx = (AddContext*)v->_ctx;
            build_topo(ctx->a, topo, n_topo);
            build_topo(ctx->b, topo, n_topo);
        } else if (v->_backward == softmax_backward_2d || v->_backward == softmax_backward_3d) {
            SoftmaxContext* ctx = (SoftmaxContext*)v->_ctx;
            build_topo(ctx->input, topo, n_topo);
        } else if (v->_backward == linear_backward_2d || v->_backward == linear_backward_3d) {
            LinearContext* ctx = (LinearContext*)v->_ctx;
            build_topo(ctx->input, topo, n_topo);
        } else if (v->_backward == layernorm_backward_2d || v->_backward == layernorm_backward_3d) {
            LayerNormContext* ctx = (LayerNormContext*)v->_ctx;
            build_topo(ctx->input, topo, n_topo);
        } else if (v->_backward == matmul_backward) {
            MatmulContext* ctx = (MatmulContext*)v->_ctx;
            build_topo(ctx->a, topo, n_topo);
            build_topo(ctx->b, topo, n_topo);
        } else if (v->_backward == transpose_backward) {
            TransposeContext* ctx = (TransposeContext*)v->_ctx;
            build_topo(ctx->input, topo, n_topo);
        } else if (v->_backward == head_backward) {
            HeadContext* ctx = (HeadContext*)v->_ctx;
            build_topo(ctx->x, topo, n_topo);
        } else if (v->_backward == mha_backward) {
            MultiHeadAttentionContext* ctx = (MultiHeadAttentionContext*)v->_ctx;
            build_topo(ctx->x, topo, n_topo);
        } else if (v->_backward == gelu_backward) {
            GELUContext* ctx = (GELUContext*)v->_ctx;
            build_topo(ctx->input, topo, n_topo);
        } else if (v->_backward == mlp_backward) {
            MLPContext* ctx = (MLPContext*)v->_ctx;
            build_topo(ctx->x, topo, n_topo);
        } else if (v->_backward == cross_entropy_backward) {
            CrossEntropyContext* ctx = (CrossEntropyContext*)v->_ctx;
            build_topo(ctx->logits, topo, n_topo);
        } else if (v->_backward == embedding_backward_1d || v->_backward == embedding_backward_2d) {
            // Embedding has no tensor inputs
        }
    }
    
    topo[(*n_topo)++] = v;
}


/*
  Perform backpropagation starting from a given tensor.
*/
static inline void backward(Tensor* t) {
    if (t->grad == NULL) {
        printf("backward: ERROR: grad is NULL\n");
        return;
    }

    // Fill gradient with 1s
    int n = tensor_numel(t);
    for (int i = 0; i < n; ++i) {
        t->grad[i] = 1.0f;
    }

    // Build topological sort
    Tensor* topo[1000]; // Assuming max 1000 tensors in the graph
    int n_topo = 0;
    build_topo(t, topo, &n_topo);

    // Go one-by-one and backprop
    for (int i = n_topo - 1; i >= 0; --i) {
        if (topo[i]->_backward != NULL) {
            topo[i]->_backward(topo[i]);
        }
        topo[i]->visited = 0; // Reset visited flag
    }
}
