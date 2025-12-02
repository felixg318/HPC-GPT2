// cross_entropy.h
// Cross-entropy loss for logits of shape (B, T, V) and integer targets.

#pragma once

#include <math.h>
#include <stdio.h>
#include "tensor.h"
#include "softmax.h"

// Context for the cross-entropy operation
typedef struct {
    Tensor* logits;
    const int* targets;
    int B;
    int T;
} CrossEntropyContext;

// Backward function for cross-entropy loss
static inline void cross_entropy_backward(Tensor* t) {
    CrossEntropyContext* ctx = (CrossEntropyContext*)t->_ctx;
    Tensor* logits = ctx->logits;
    const int* targets = ctx->targets;
    int B = ctx->B;
    int T = ctx->T;
    int V = logits->shape[2];
    int N = B * T;

    // The gradient of the cross entropy loss is softmax(logits) - y
    // where y is the one-hot encoded target.
    
    Tensor probs;
    softmax_forward(logits, &probs);

    int used_cuda = 0;
#ifdef USE_CUDA
    int total = B * T;
    int* d_targets = NULL;
    if (CUDA_CHECK(cudaMallocManaged((void**)&d_targets, total * sizeof(int)))) {
        memcpy(d_targets, targets, total * sizeof(int));
        used_cuda = cuda_cross_entropy_backward_3d(probs.data, d_targets, logits->grad, B, T, V);
        CUDA_CHECK(cudaFree(d_targets));
    }
#endif

    if (!used_cuda) {
        for (int b = 0; b < B; ++b) {
            for (int t_ = 0; t_ < T; ++t_) {
                int target_idx = targets[b * T + t_];
                for (int v = 0; v < V; ++v) {
                    float prob = tensor_get3(&probs, b, t_, v);
                    float grad = prob - (v == target_idx ? 1.0f : 0.0f);
                    logits->grad[tensor_index3(logits, b, t_, v)] += grad / N;
                }
            }
        }
    }
    
    tensor_free(&probs);
    free(ctx);
}


/*
  Compute mean cross-entropy loss over logits and targets.

  logits : Tensor of shape (B, T, V)
  targets: int array of shape (B * T), flattened as targets[b*T + t]
           each entry in [0, V)

  Returns:
    scalar float loss = average over all (b,t).

  NOTE:
    - No ignore_index.
    - No masking.
    - Just plain mean over all positions.
*/
static inline void cross_entropy_loss_3d(const Tensor* logits,
                                          const int* targets,
                                          int B,
                                          int T,
                                          Tensor* out) {
    if (logits->ndim != 3) {
        printf("cross_entropy_loss_3d: ERROR: logits->ndim must be 3 (B,T,V)\n");
        return;
    }

    int V = logits->shape[2];

    // Sanity check
    if (logits->shape[0] != B || logits->shape[1] != T) {
        printf("cross_entropy_loss_3d: ERROR: shape mismatch: logits=(%d,%d,%d), B=%d, T=%d\n",
               logits->shape[0], logits->shape[1], logits->shape[2], B, T);
        return;
    }

    int N = B * T;  // total number of positions
    float loss_sum = 0.0f;

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            int idx_target = targets[b * T + t];  // target class for this position

            if (idx_target < 0 || idx_target >= V) {
                printf("cross_entropy_loss_3d: WARNING: target=%d out of range [0,%d)\n",
                       idx_target, V);
                continue;
            }

            // 1) find max logit for numerical stability
            float max_logit = -1e30f;
            for (int v = 0; v < V; ++v) {
                float z = tensor_get3(logits, b, t, v);
                if (z > max_logit) max_logit = z;
            }

            // 2) compute log-sum-exp
            float sum_exp = 0.0f;
            for (int v = 0; v < V; ++v) {
                float z = tensor_get3(logits, b, t, v);
                sum_exp += expf(z - max_logit);
            }
            float log_sum_exp = max_logit + logf(sum_exp);

            // 3) contribution of the correct class
            float z_y = tensor_get3(logits, b, t, idx_target);
            float log_prob = z_y - log_sum_exp;   // log softmax
            float ce = -log_prob;                 // cross-entropy term

            loss_sum += ce;
        }
    }

    float mean_loss = loss_sum / (float)N;
    
    int out_shape[1] = {1};
    tensor_init(out, 1, out_shape);
    out->data[0] = mean_loss;

    // Create context for autograd
    CrossEntropyContext* ctx = (CrossEntropyContext*)malloc(sizeof(CrossEntropyContext));
    ctx->logits = (Tensor*)logits;
    ctx->targets = targets;
    ctx->B = B;
    ctx->T = T;
    
    tensor_set_inputs1(out, (Tensor*)logits);
    out->_ctx = ctx;
    out->_backward = cross_entropy_backward;
}
