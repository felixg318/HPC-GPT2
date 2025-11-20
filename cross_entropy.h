// cross_entropy.h
// Cross-entropy loss for logits of shape (B, T, V) and integer targets.

#pragma once

#include <math.h>
#include <stdio.h>
#include "tensor.h"

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
static inline float cross_entropy_loss_3d(const Tensor* logits,
                                          const int* targets,
                                          int B,
                                          int T) {
    if (logits->ndim != 3) {
        printf("cross_entropy_loss_3d: ERROR: logits->ndim must be 3 (B,T,V)\n");
        return 0.0f;
    }

    int V = logits->shape[2];

    // Sanity check
    if (logits->shape[0] != B || logits->shape[1] != T) {
        printf("cross_entropy_loss_3d: ERROR: shape mismatch: logits=(%d,%d,%d), B=%d, T=%d\n",
               logits->shape[0], logits->shape[1], logits->shape[2], B, T);
        return 0.0f;
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
    return mean_loss;
}

