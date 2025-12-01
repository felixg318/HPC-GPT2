// adam.h
// Adam optimizer.

#pragma once

#include <stdio.h>
#include <math.h>
#include <mpi.h>
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
    int rank;
    int world_size;
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
    optimizer->rank = 0;
    optimizer->world_size = 1;

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

static inline void adam_set_distributed(AdamOptimizer* optimizer, int rank, int world_size) {
    optimizer->rank = rank;
    optimizer->world_size = (world_size > 0) ? world_size : 1;
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
        for (int j = 0; j < n; ++j) {
            float grad = param->grad[j];
            
            // Update biased first moment estimate
            m->data[j] = optimizer->beta1 * m->data[j] + (1.0f - optimizer->beta1) * grad;
            
            // Update biased second raw moment estimate
            v->data[j] = optimizer->beta2 * v->data[j] + (1.0f - optimizer->beta2) * grad * grad;
            
            // Update parameters
            param->data[j] -= lr_t * m->data[j] / (sqrt(v->data[j]) + optimizer->eps);
        }
        if (optimizer->world_size > 1) {
            MPI_Bcast(param->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(m->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(v->data, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
