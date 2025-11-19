// linear.h
// Simple Linear layer: y = x W^T + b, using Tensor from tensor.h

#pragma once

#include "tensor.h"

/*
  Linear layer struct.

  - in_features:  input dimension C_in
  - out_features: output dimension C_out
  - use_bias:     whether to use bias or not (1 = yes, 0 = no)
  - weight:       shape (out_features, in_features)
  - bias:         shape (out_features)  if use_bias == 1
*/
typedef struct {
    int in_features;
    int out_features;
    int use_bias;
    Tensor weight;
    Tensor bias;
} Linear;

/*
  Initialize a Linear layer.

  This allocates memory for:
    - weight (out_features, in_features)
    - bias (out_features) if use_bias == 1

  NOTE: It does NOT initialize the values to anything special.
        You can call tensor_fill(&layer.weight, value) if you want.
*/
static inline void linear_init(Linear* layer, int in_f, int out_f, int use_bias) {
    layer->in_features  = in_f;
    layer->out_features = out_f;
    layer->use_bias     = use_bias;

    int w_shape[2];
    w_shape[0] = out_f;
    w_shape[1] = in_f;
    tensor_init(&layer->weight, 2, w_shape);  // (out_f, in_f)

    if (use_bias) {
        int b_shape[1];
        b_shape[0] = out_f;
        tensor_init(&layer->bias, 1, b_shape);  // (out_f)
    } else {
        layer->bias.data   = NULL;
        layer->bias.ndim   = 0;
        layer->bias.shape[0] = 0;
    }
}

/*
  Free the memory used by the Linear layer (weights and bias).
*/
static inline void linear_free(Linear* layer) {
    tensor_free(&layer->weight);
    if (layer->use_bias) {
        tensor_free(&layer->bias);
    }
}

/*
  Forward for 2D input: x shape = (N, in_features)
  Output y shape = (N, out_features)

  Computes:
    y[n, o] = sum over c ( W[o, c] * x[n, c] ) + bias[o]
*/
static inline void linear_forward2D(const Linear* layer, const Tensor* x, Tensor* y) {
    // Check that x is 2D
    if (x->ndim != 2) {
        printf("linear_forward2D: ERROR: x->ndim must be 2\n");
        return;
    }

    int N = x->shape[0];
    int C = x->shape[1];

    if (C != layer->in_features) {
        printf("linear_forward2D: ERROR: input features mismatch: C=%d, expected %d\n",
               C, layer->in_features);
        return;
    }

    // Initialize output tensor y with shape (N, out_features)
    int y_shape[2];
    y_shape[0] = N;
    y_shape[1] = layer->out_features;
    tensor_init(y, 2, y_shape);

    // Compute y[n, o]
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < layer->out_features; ++o) {
            float sum = 0.0f;

            // Dot product between x[n, :] and weight[o, :]
            for (int c = 0; c < layer->in_features; ++c) {
                float w_oc = tensor_get2(&layer->weight, o, c); // W[o, c]
                float x_nc = tensor_get2(x, n, c);              // x[n, c]
                sum += w_oc * x_nc;
            }

            // Add bias if used
            if (layer->use_bias) {
                float b_o = tensor_get1(&layer->bias, o);
                sum += b_o;
            }

            tensor_set2(y, n, o, sum);
        }
    }
}

/*
  Forward for 3D input: x shape = (B, T, in_features)
  Output y shape = (B, T, out_features)

  Computes:
    y[b, t, o] = sum over c ( W[o, c] * x[b, t, c] ) + bias[o]
*/
static inline void linear_forward3D(const Linear* layer, const Tensor* x, Tensor* y) {
    if (x->ndim != 3) {
        printf("linear_forward3D: ERROR: x->ndim must be 3\n");
        return;
    }

    int B = x->shape[0];
    int T = x->shape[1];
    int C = x->shape[2];

    if (C != layer->in_features) {
        printf("linear_forward3D: ERROR: input features mismatch: C=%d, expected %d\n",
               C, layer->in_features);
        return;
    }

    // Initialize output tensor y with shape (B, T, out_features)
    int y_shape[3];
    y_shape[0] = B;
    y_shape[1] = T;
    y_shape[2] = layer->out_features;
    tensor_init(y, 3, y_shape);

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < layer->out_features; ++o) {
                float sum = 0.0f;

                // Dot product between x[b, t, :] and weight[o, :]
                for (int c = 0; c < layer->in_features; ++c) {
                    float w_oc = tensor_get2(&layer->weight, o, c);   // W[o, c]
                    float x_btc = tensor_get3(x, b, t, c);            // x[b, t, c]
                    sum += w_oc * x_btc;
                }

                if (layer->use_bias) {
                    float b_o = tensor_get1(&layer->bias, o);
                    sum += b_o;
                }

                tensor_set3(y, b, t, o, sum);
            }
        }
    }
}

/*
  Convenience wrapper that calls either 2D or 3D forward
  depending on x->ndim.
*/
static inline void linear_forward(const Linear* layer, const Tensor* x, Tensor* y) {
    if (x->ndim == 2) {
        linear_forward2D(layer, x, y);
    } else if (x->ndim == 3) {
        linear_forward3D(layer, x, y);
    } else {
        printf("linear_forward: ERROR: x->ndim must be 2 or 3, got %d\n", x->ndim);
    }
}

