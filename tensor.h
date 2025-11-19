// tensor.h
// Minimal tensor struct using raw pointers (no std::vector).

#pragma once   // Make sure the header is only included once per translation unit

#include <stdlib.h>   // for malloc, free
#include <stdio.h>    // for printf (optional, for error messages)

#define TENSOR_MAX_DIMS 4  // we only need up to 3D (plus a little safety)

/*
  A very simple N-dimensional tensor of floats.

  - data: pointer to a contiguous block of memory
  - shape: sizes of each dimension, e.g. shape[0]=B, shape[1]=T, shape[2]=C
  - ndim: how many dimensions are actually used (1,2,3,...)
*/
typedef struct {
    float* data;                    // pointer to data on CPU
    int    shape[TENSOR_MAX_DIMS];  // sizes of each dimension
    int    ndim;                    // number of dimensions actually used
} Tensor;

/*
  Helper: compute number of elements = shape[0] * shape[1] * ... * shape[ndim-1]
*/
static inline int tensor_numel(const Tensor* t) {
    int n = 1;
    for (int i = 0; i < t->ndim; ++i) {
        n *= t->shape[i];
    }
    return n;
}

/*
  Initialize a tensor with given shape.

  - t: pointer to Tensor struct
  - ndim: number of dimensions (1, 2, or 3 for our use)
  - shape: pointer to an array of ints of length ndim

  This will allocate memory using malloc.
*/
static inline void tensor_init(Tensor* t, int ndim, const int* shape) {
    t->ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    // zero out remaining shape entries just for safety
    for (int i = ndim; i < TENSOR_MAX_DIMS; ++i) {
        t->shape[i] = 0;
    }

    int n = tensor_numel(t);
    t->data = (float*)malloc(n * sizeof(float));
    if (t->data == NULL) {
        printf("tensor_init: ERROR: malloc failed for %d elements\n", n);
    }
}

/*
  Free the memory owned by this tensor.
*/
static inline void tensor_free(Tensor* t) {
    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }
    t->ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
        t->shape[i] = 0;
    }
}

/*
  Fill the entire tensor with a constant value.
*/
static inline void tensor_fill(Tensor* t, float value) {
    int n = tensor_numel(t);
    for (int i = 0; i < n; ++i) {
        t->data[i] = value;
    }
}

/*
  Index helpers.

  We store the data in "row-major" order, meaning:

    For 2D: index = i * shape[1] + j
    For 3D: index = (i * shape[1] + j) * shape[2] + k

  We expose helper functions to compute these indices and get/set values.

  NOTE: We do NOT check bounds here to keep it simple and fast.
*/

/* 1D: t->shape = {N}, ndim=1 */
static inline int tensor_index1(const Tensor* t, int i) {
    // i in [0, shape[0])
    return i;
}

/* 2D: t->shape = {N, C}, ndim=2 */
static inline int tensor_index2(const Tensor* t, int i, int j) {
    int C = t->shape[1];
    return i * C + j;
}

/* 3D: t->shape = {B, T, C}, ndim=3 */
static inline int tensor_index3(const Tensor* t, int i, int j, int k) {
    int T = t->shape[1];
    int C = t->shape[2];
    return (i * T + j) * C + k;
}

/*
  Get / set helpers for common dimensions.
  These just use the index functions above.
*/

// 1D get/set
static inline float tensor_get1(const Tensor* t, int i) {
    int idx = tensor_index1(t, i);
    return t->data[idx];
}

static inline void tensor_set1(Tensor* t, int i, float value) {
    int idx = tensor_index1(t, i);
    t->data[idx] = value;
}

// 2D get/set
static inline float tensor_get2(const Tensor* t, int i, int j) {
    int idx = tensor_index2(t, i, j);
    return t->data[idx];
}

static inline void tensor_set2(Tensor* t, int i, int j, float value) {
    int idx = tensor_index2(t, i, j);
    t->data[idx] = value;
}

// 3D get/set
static inline float tensor_get3(const Tensor* t, int i, int j, int k) {
    int idx = tensor_index3(t, i, j, k);
    return t->data[idx];
}

static inline void tensor_set3(Tensor* t, int i, int j, int k, float value) {
    int idx = tensor_index3(t, i, j, k);
    t->data[idx] = value;
}

