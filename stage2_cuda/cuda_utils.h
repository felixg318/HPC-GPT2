// cuda_utils.h
// Lightweight CUDA helpers and host-callable wrappers.

#pragma once

#include <stdbool.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdio.h>

#ifndef CUDA_USE_MANAGED
#define CUDA_USE_MANAGED 1
#endif

static inline bool cuda_check_internal(cudaError_t err,
                                       const char* expr,
                                       const char* file,
                                       int line) {
    if (err != cudaSuccess) {
        fprintf(stderr,
                "CUDA error at %s:%d for %s: %s\n",
                file,
                line,
                expr,
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

#define CUDA_CHECK(expr) cuda_check_internal((expr), #expr, __FILE__, __LINE__)

#ifdef __cplusplus
extern "C" {
#endif

// Matrix multiplication (host pointers):
// 2D: A (N, C1) @ B (C1, C2) -> C (N, C2)
bool cuda_matmul_2d(const float* A, const float* B, float* C,
                    int N, int C1, int C2);

// 3D batch: A (B, T, C1) @ B (B, C1, C2) -> C (B, T, C2)
bool cuda_matmul_3d(const float* A, const float* B, float* C, int Bdim, int T, int C1, int C2);

// Gradients for matmul
bool cuda_matmul_2d_backward(const float* grad_C,
                             const float* A,
                             const float* B,
                             float* grad_A,
                             float* grad_B,
                             int N, int C1, int C2);

bool cuda_matmul_3d_backward(const float* grad_C,
                             const float* A,
                             const float* B,
                             float* grad_A,
                             float* grad_B,
                             int Bdim, int T, int C1, int C2);

// Softmax over last dim for (N, C)
bool cuda_softmax_2d(const float* X, float* Y, int N, int C);
bool cuda_softmax_2d_backward(const float* Y, const float* grad_Y, float* grad_X, int N, int C);

// Softmax over last dim for (B, T, C)
bool cuda_softmax_3d(const float* X, float* Y, int B, int T, int C);
bool cuda_softmax_3d_backward(const float* Y, const float* grad_Y, float* grad_X, int B, int T, int C);

// GELU (approximate) elementwise on contiguous buffer of length n
bool cuda_gelu(const float* X, float* Y, int n);
bool cuda_gelu_backward(const float* X, const float* grad_Y, float* grad_X, int n);

// LayerNorm over last dim C
bool cuda_layernorm_2d(const float* X, float* Y,
                       const float* gamma, const float* beta,
                       int N, int C, float eps);
bool cuda_layernorm_3d(const float* X, float* Y,
                       const float* gamma, const float* beta,
                       int B, int T, int C, float eps);
bool cuda_layernorm_2d_backward(const float* X, const float* grad_Y,
                                const float* gamma,
                                float* grad_X, float* grad_gamma, float* grad_beta,
                                int N, int C, float eps);
bool cuda_layernorm_3d_backward(const float* X, const float* grad_Y,
                                const float* gamma,
                                float* grad_X, float* grad_gamma, float* grad_beta,
                                int B, int T, int C, float eps);

// Embedding backward scatter for (B,T)
bool cuda_embedding_backward_2d(const int* idx, const float* grad_out,
                                float* grad_weight,
                                int B, int T, int vocab_size, int dim);

// Cross-entropy grad for logits (B,T,V) using precomputed softmax probs
bool cuda_cross_entropy_backward_3d(const float* probs, const int* targets,
                                    float* grad_logits,
                                    int B, int T, int V);

// Adam parameter update: in-place on param/m/v/grad
bool cuda_adam_step(float* param, float* grad, float* m, float* v,
                    int n, float lr_t, float beta1, float beta2, float eps);

// Causal mask: set att[b,tq,tk] = -1e30 when tk > tq for (B,T,T)
bool cuda_apply_causal_mask(float* att, int B, int T);

// Copy one head output (B,T,head_size) into concat (B,T,n_embd) at offset head_idx*head_size
bool cuda_concat_head(const float* head, float* concat,
                      int B, int T, int head_size, int head_idx, int n_heads);

// Transpose last two dims of a 3D tensor (B, T1, T2) -> (B, T2, T1)
bool cuda_transpose_last2(const float* X, float* Y, int B, int T1, int T2);
bool cuda_transpose_last2_backward(const float* grad_Y, float* grad_X, int B, int T1, int T2);

// Zero a contiguous float buffer of length n
bool cuda_zero_buffer(float* data, int n);

// Grad norm accumulation and scaling helpers
bool cuda_grad_norm_accum(const float* grad, int n, float* sum);
bool cuda_grad_scale(float* grad, int n, float scale);

#ifdef __cplusplus
}
#endif

#else

// Stub macros and functions when CUDA is not enabled at compile time.
#define CUDA_CHECK(expr) (true)

static inline bool cuda_matmul_2d(const float* A, const float* B, float* C,
                                  int N, int C1, int C2) {
    (void)A; (void)B; (void)C; (void)N; (void)C1; (void)C2;
    return false;
}

static inline bool cuda_matmul_3d(const float* A, const float* B, float* C,
                                  int Bdim, int T, int C1, int C2) {
    (void)A; (void)B; (void)C; (void)Bdim; (void)T; (void)C1; (void)C2;
    return false;
}

static inline bool cuda_matmul_2d_backward(const float* grad_C,
                                           const float* A,
                                           const float* B,
                                           float* grad_A,
                                           float* grad_B,
                                           int N, int C1, int C2) {
    (void)grad_C; (void)A; (void)B; (void)grad_A; (void)grad_B; (void)N; (void)C1; (void)C2;
    return false;
}

static inline bool cuda_matmul_3d_backward(const float* grad_C,
                                           const float* A,
                                           const float* B,
                                           float* grad_A,
                                           float* grad_B,
                                           int Bdim, int T, int C1, int C2) {
    (void)grad_C; (void)A; (void)B; (void)grad_A; (void)grad_B; (void)Bdim; (void)T; (void)C1; (void)C2;
    return false;
}

static inline bool cuda_softmax_2d(const float* X, float* Y, int N, int C) {
    (void)X; (void)Y; (void)N; (void)C;
    return false;
}

static inline bool cuda_softmax_2d_backward(const float* Y, const float* grad_Y, float* grad_X, int N, int C) {
    (void)Y; (void)grad_Y; (void)grad_X; (void)N; (void)C;
    return false;
}

static inline bool cuda_softmax_3d(const float* X, float* Y, int B, int T, int C) {
    (void)X; (void)Y; (void)B; (void)T; (void)C;
    return false;
}

static inline bool cuda_softmax_3d_backward(const float* Y, const float* grad_Y, float* grad_X, int B, int T, int C) {
    (void)Y; (void)grad_Y; (void)grad_X; (void)B; (void)T; (void)C;
    return false;
}

static inline bool cuda_gelu(const float* X, float* Y, int n) {
    (void)X; (void)Y; (void)n;
    return false;
}

static inline bool cuda_gelu_backward(const float* X, const float* grad_Y, float* grad_X, int n) {
    (void)X; (void)grad_Y; (void)grad_X; (void)n;
    return false;
}

static inline bool cuda_layernorm_2d(const float* X, float* Y,
                                     const float* gamma, const float* beta,
                                     int N, int C, float eps) {
    (void)X; (void)Y; (void)gamma; (void)beta; (void)N; (void)C; (void)eps;
    return false;
}

static inline bool cuda_layernorm_3d(const float* X, float* Y,
                                     const float* gamma, const float* beta,
                                     int B, int T, int C, float eps) {
    (void)X; (void)Y; (void)gamma; (void)beta; (void)B; (void)T; (void)C; (void)eps;
    return false;
}

static inline bool cuda_layernorm_2d_backward(const float* X, const float* grad_Y,
                                              const float* gamma,
                                              float* grad_X, float* grad_gamma, float* grad_beta,
                                              int N, int C, float eps) {
    (void)X; (void)grad_Y; (void)gamma; (void)grad_X; (void)grad_gamma; (void)grad_beta; (void)N; (void)C; (void)eps;
    return false;
}

static inline bool cuda_layernorm_3d_backward(const float* X, const float* grad_Y,
                                              const float* gamma,
                                              float* grad_X, float* grad_gamma, float* grad_beta,
                                              int B, int T, int C, float eps) {
    (void)X; (void)grad_Y; (void)gamma; (void)grad_X; (void)grad_gamma; (void)grad_beta; (void)B; (void)T; (void)C; (void)eps;
    return false;
}

static inline bool cuda_embedding_backward_2d(const int* idx, const float* grad_out,
                                              float* grad_weight,
                                              int B, int T, int vocab_size, int dim) {
    (void)idx; (void)grad_out; (void)grad_weight; (void)B; (void)T; (void)vocab_size; (void)dim;
    return false;
}

static inline bool cuda_cross_entropy_backward_3d(const float* probs, const int* targets,
                                                  float* grad_logits,
                                                  int B, int T, int V) {
    (void)probs; (void)targets; (void)grad_logits; (void)B; (void)T; (void)V;
    return false;
}

static inline bool cuda_adam_step(float* param, float* grad, float* m, float* v,
                                  int n, float lr_t, float beta1, float beta2, float eps) {
    (void)param; (void)grad; (void)m; (void)v; (void)n; (void)lr_t; (void)beta1; (void)beta2; (void)eps;
    return false;
}

static inline bool cuda_apply_causal_mask(float* att, int B, int T) {
    (void)att; (void)B; (void)T;
    return false;
}

static inline bool cuda_concat_head(const float* head, float* concat,
                                    int B, int T, int head_size, int head_idx, int n_heads) {
    (void)head; (void)concat; (void)B; (void)T; (void)head_size; (void)head_idx; (void)n_heads;
    return false;
}

static inline bool cuda_transpose_last2(const float* X, float* Y, int B, int T1, int T2) {
    (void)X; (void)Y; (void)B; (void)T1; (void)T2;
    return false;
}

static inline bool cuda_transpose_last2_backward(const float* grad_Y, float* grad_X, int B, int T1, int T2) {
    (void)grad_Y; (void)grad_X; (void)B; (void)T1; (void)T2;
    return false;
}

static inline bool cuda_zero_buffer(float* data, int n) {
    (void)data; (void)n;
    return false;
}

static inline bool cuda_grad_norm_accum(const float* grad, int n, float* sum) {
    (void)grad; (void)n; (void)sum;
    return false;
}

static inline bool cuda_grad_scale(float* grad, int n, float scale) {
    (void)grad; (void)n; (void)scale;
    return false;
}

#endif  // USE_CUDA
