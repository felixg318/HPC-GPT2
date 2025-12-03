// adam.h
// Adam optimizer.

#pragma once

#include <stdio.h>
#include <math.h>
#include "tensor.h"

// Forward declaration of AdamOptimizer to use it in function pointers.
struct AdamOptimizer;

typedef struct AdamOptimizer {
    Tensor** params;       // Array of pointers to model parameters
    int num_params;        // Number of parameters
    Tensor* m;             // First moment vectors
    Tensor* v;             // Second moment vectors
    int t;                 // Timestep
    float lr;              // Learning rate
    float beta1;           // Exponential decay rate for the first moment estimates
    float beta2;           // Exponential decay rate for the second moment estimates
    float eps;             // A small constant for numerical stability
    void (*lr_scheduler)(struct AdamOptimizer* optimizer); // Learning rate scheduler
} AdamOptimizer;

/*
  Initialize the Adam optimizer.

  Arguments:
    optimizer   : pointer to AdamOptimizer
    params      : array of pointers to model parameters
    num_params  : number of parameters
    lr          : learning rate
    beta1       : exponential decay rate for the first moment estimates
    beta2       : exponential decay rate for the second moment estimates
    eps         : a small constant for numerical stability
*/
static inline void adam_init(AdamOptimizer* optimizer, Tensor** params, int num_params, float lr, float beta1, float beta2, float eps) {
    optimizer->params = params;
    optimizer->num_params = num_params;
    optimizer->lr = lr;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->eps = eps;
    optimizer->t = 0;
    optimizer->lr_scheduler = NULL;

    optimizer->m = (Tensor*)malloc(num_params * sizeof(Tensor));
    optimizer->v = (Tensor*)malloc(num_params * sizeof(Tensor));

    for (int i = 0; i < num_params; ++i) {
        tensor_init(&optimizer->m[i], params[i]->ndim, params[i]->shape);
        tensor_init(&optimizer->v[i], params[i]->ndim, params[i]->shape);
        tensor_zero(&optimizer->m[i]);
        tensor_zero(&optimizer->v[i]);
    }
}

/*
  Free resources in an AdamOptimizer.
*/
static inline void adam_free(AdamOptimizer* optimizer) {
    for (int i = 0; i < optimizer->num_params; ++i) {
        tensor_free(&optimizer->m[i]);
        tensor_free(&optimizer->v[i]);
    }
    free(optimizer->m);
    free(optimizer->v);
}

/*
  Zero out the gradients of all parameters.
*/
static inline void adam_zero_grad(AdamOptimizer* optimizer) {
    for (int i = 0; i < optimizer->num_params; ++i) {
        tensor_zero_grad(optimizer->params[i]);
    }
}

/*
  Clip the gradients to a maximum norm.
*/
static inline void clip_grad_norm(AdamOptimizer* optimizer, float max_norm) {
#ifdef USE_CUDA
    float* d_sum = NULL;
    if (CUDA_CHECK(cudaMallocManaged((void**)&d_sum, sizeof(float)))) {
        *d_sum = 0.0f;
        // accumulate norm
        for (int i = 0; i < optimizer->num_params; ++i) {
            Tensor* p = optimizer->params[i];
            int n = tensor_numel(p);
            if (!cuda_grad_norm_accum(p->grad, n, d_sum)) {
                // fallback to CPU for this tensor
                for (int j = 0; j < n; ++j) {
                    *d_sum += p->grad[j] * p->grad[j];
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        float total_norm = sqrtf(*d_sum);
        if (total_norm > max_norm) {
            float scale = max_norm / total_norm;
            for (int i = 0; i < optimizer->num_params; ++i) {
                Tensor* p = optimizer->params[i];
                int n = tensor_numel(p);
                if (!cuda_grad_scale(p->grad, n, scale)) {
                    for (int j = 0; j < n; ++j) {
                        p->grad[j] *= scale;
                    }
                }
            }
        }
        CUDA_CHECK(cudaFree(d_sum));
    }
#else
    float total_norm = 0.0f;
    for (int i = 0; i < optimizer->num_params; ++i) {
        int n = tensor_numel(optimizer->params[i]);
        for (int j = 0; j < n; ++j) {
            total_norm += optimizer->params[i]->grad[j] * optimizer->params[i]->grad[j];
        }
    }
    total_norm = sqrt(total_norm);
    
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < optimizer->num_params; ++i) {
            int n = tensor_numel(optimizer->params[i]);
            for (int j = 0; j < n; ++j) {
                optimizer->params[i]->grad[j] *= scale;
            }
        }
    }
#endif
}

/*
  Perform one optimization step.
*/
static inline void adam_step(AdamOptimizer* optimizer, float clip_grad_norm_val) {
    if (optimizer->lr_scheduler != NULL) {
        optimizer->lr_scheduler(optimizer);
    }

    if (clip_grad_norm_val > 0.0f) {
        clip_grad_norm(optimizer, clip_grad_norm_val);
    }
    
    optimizer->t++;

    float lr_t = optimizer->lr * sqrt(1.0f - pow(optimizer->beta2, optimizer->t)) / (1.0f - pow(optimizer->beta1, optimizer->t));

    for (int i = 0; i < optimizer->num_params; ++i) {
        Tensor* param = optimizer->params[i];
        Tensor* m = &optimizer->m[i];
        Tensor* v = &optimizer->v[i];
        
        int n = tensor_numel(param);
        int used_cuda = 0;
    #ifdef USE_CUDA
        used_cuda = cuda_adam_step(param->data, param->grad, m->data, v->data,
                                   n, lr_t, optimizer->beta1, optimizer->beta2, optimizer->eps);
    #endif

        if (!used_cuda) {
            for (int j = 0; j < n; ++j) {
                float grad = param->grad[j];
                
                // Update biased first moment estimate
                m->data[j] = optimizer->beta1 * m->data[j] + (1.0f - optimizer->beta1) * grad;
                
                // Update biased second raw moment estimate
                v->data[j] = optimizer->beta2 * v->data[j] + (1.0f - optimizer->beta2) * grad * grad;
                
                // Update parameters
                param->data[j] -= lr_t * m->data[j] / (sqrt(v->data[j]) + optimizer->eps);
            }
        }
    }
}

// Example of a linear learning rate decay scheduler
static inline void linear_lr_decay(AdamOptimizer* optimizer) {
    // This is just an example. A real implementation would be more flexible.
    int total_steps = 10000;
    float start_lr = 1e-4;
    float end_lr = 1e-5;
    if (optimizer->t < total_steps) {
        optimizer->lr = start_lr - (start_lr - end_lr) * ((float)optimizer->t / total_steps);
    } else {
        optimizer->lr = end_lr;
    }
}
