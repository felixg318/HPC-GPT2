// cross_entropy.h
// Cross-entropy loss for logits of shape (B, T, V) and integer targets.

#pragma once

#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include "tensor.h"
#include "softmax.h"

static int g_cross_entropy_rank = 0;
static int g_cross_entropy_world = 1;

static inline void cross_entropy_set_distributed(int rank, int world_size) {
    g_cross_entropy_rank = (rank >= 0) ? rank : 0;
    g_cross_entropy_world = (world_size > 0) ? world_size : 1;
}

// Context for the cross-entropy operation
typedef struct {
    Tensor* logits;
    const int* targets;
    int B;
    int T;
    float inv_token_count;
} CrossEntropyContext;

// Backward function for cross-entropy loss
static inline void cross_entropy_backward(Tensor* t) {
    CrossEntropyContext* ctx = (CrossEntropyContext*)t->_ctx;
    Tensor* logits = ctx->logits;
    const int* targets = ctx->targets;
    int B = ctx->B;
    int T = ctx->T;
    int V = logits->shape[2];

    // The gradient of the cross entropy loss is softmax(logits) - y
    // where y is the one-hot encoded target.
    
    Tensor probs;
    softmax_forward(logits, &probs);

    for (int b = 0; b < B; ++b) {
        for (int t_ = 0; t_ < T; ++t_) {
            int target_idx = targets[b * T + t_];
            for (int v = 0; v < V; ++v) {
                float prob = tensor_get3(&probs, b, t_, v);
                float grad = prob - (v == target_idx ? 1.0f : 0.0f);
                logits->grad[tensor_index3(logits, b, t_, v)] += grad * ctx->inv_token_count;
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

    double accum[2];
    accum[0] = (double)loss_sum;
    accum[1] = (double)N;
    if (g_cross_entropy_world > 1) {
        MPI_Allreduce(MPI_IN_PLACE, accum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    float mean_loss = (float)(accum[0] / (accum[1] > 0.0 ? accum[1] : 1.0));
    float inv_token_count = (float)(1.0 / (accum[1] > 0.0 ? accum[1] : 1.0));
    
    int out_shape[1] = {1};
    tensor_init(out, 1, out_shape);
    out->data[0] = mean_loss;

    // Create context for autograd
    CrossEntropyContext* ctx = (CrossEntropyContext*)malloc(sizeof(CrossEntropyContext));
    ctx->logits = (Tensor*)logits;
    ctx->targets = targets;
    ctx->B = B;
    ctx->T = T;
    ctx->inv_token_count = inv_token_count;
    
    tensor_set_inputs1(out, (Tensor*)logits);
    out->_ctx = ctx;
    out->_backward = cross_entropy_backward;
}
