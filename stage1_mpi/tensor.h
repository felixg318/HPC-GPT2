// tensor.h
// Minimal tensor struct using raw pointers (no std::vector).

#pragma once   // Make sure the header is only included once per translation unit

#include <stdlib.h>   // for malloc, free, realloc
#include <stdio.h>    // for printf (optional, for error messages)
#include <string.h>   // for memset
#include <math.h>

#define TENSOR_MAX_DIMS 4  // we only need up to 3D (plus a little safety)

// Forward declaration of Tensor struct to use it in function pointers.
struct Tensor;

typedef enum {
    TENSOR_DIST_REPLICATED = 0,
    TENSOR_DIST_SHARDED = 1
} TensorDistType;

/*
  A very simple N-dimensional tensor of floats.

  - data: pointer to a contiguous block of memory
  - grad: pointer to a contiguous block of memory for gradients
  - shape: sizes of each dimension, e.g. shape[0]=B, shape[1]=T, shape[2]=C
  - ndim: how many dimensions are actually used (1,2,3,...)
  - _ctx: context for autograd
  - _backward: backward function for autograd
*/
typedef struct Tensor {
    float* data;                    // pointer to data on CPU
    float* grad;                    // pointer to gradient on CPU
    int    shape[TENSOR_MAX_DIMS];  // sizes of each dimension
    int    ndim;                    // number of dimensions actually used
    TensorDistType dist_type;       // replicated vs sharded parameter
    int    dist_owner;              // optional owner rank for sharded params
    void*  _ctx;                    // autograd context
    void   (*_backward)(struct Tensor* t); // autograd backward function
    struct Tensor** _inputs;        // upstream tensors
    int    _num_inputs;             // number of upstream tensors
    int    visited;                 // for topological sort
} Tensor;

static inline void tensor_set_dist_type(Tensor* t, TensorDistType type, int owner_rank) {
    if (t == NULL) return;
    t->dist_type = type;
    t->dist_owner = owner_rank;
}

static inline int tensor_is_replicated(const Tensor* t) {
    if (t == NULL) return 1;
    return t->dist_type == TENSOR_DIST_REPLICATED;
}

static inline int tensor_is_sharded(const Tensor* t) {
    if (t == NULL) return 0;
    return t->dist_type == TENSOR_DIST_SHARDED;
}

typedef struct {
    Tensor** tensors;
    int count;
    int capacity;
} TensorTracker;

typedef struct {
    Tensor** data;
    int count;
    int capacity;
} TensorPtrArray;

static inline void tensor_free(Tensor* t);

static inline void tensor_tracker_init(TensorTracker* tracker) {
    tracker->tensors = NULL;
    tracker->count = 0;
    tracker->capacity = 0;
}

static inline void tensor_tracker_add(TensorTracker* tracker, Tensor* t) {
    if (tracker == NULL) {
        return;
    }
    if (tracker->count == tracker->capacity) {
        int new_capacity = (tracker->capacity == 0) ? 128 : tracker->capacity * 2;
        Tensor** new_list = (Tensor**)realloc(tracker->tensors, new_capacity * sizeof(Tensor*));
        if (new_list == NULL) {
            printf("tensor_tracker_add: ERROR: realloc failed\n");
            return;
        }
        tracker->tensors = new_list;
        tracker->capacity = new_capacity;
    }
    tracker->tensors[tracker->count++] = t;
}

static inline Tensor* tensor_tracker_new(TensorTracker* tracker) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if (t == NULL) {
        printf("tensor_tracker_new: ERROR: malloc failed\n");
        return NULL;
    }
    memset(t, 0, sizeof(Tensor));
    if (tracker != NULL) {
        tensor_tracker_add(tracker, t);
    }
    return t;
}

static inline void tensor_tracker_reset(TensorTracker* tracker) {
    if (tracker == NULL) {
        return;
    }
    for (int i = 0; i < tracker->count; ++i) {
        tensor_free(tracker->tensors[i]);
        free(tracker->tensors[i]);
        tracker->tensors[i] = NULL;
    }
    tracker->count = 0;
}

static inline void tensor_tracker_free(TensorTracker* tracker) {
    if (tracker == NULL) {
        return;
    }
    tensor_tracker_reset(tracker);
    if (tracker->tensors != NULL) {
        free(tracker->tensors);
        tracker->tensors = NULL;
    }
    tracker->capacity = 0;
}

static inline void tensor_tracker_release(TensorTracker* tracker, Tensor* t) {
    if (tracker == NULL && t != NULL) {
        tensor_free(t);
        free(t);
    }
}

static inline void tensor_ptr_array_init(TensorPtrArray* arr) {
    arr->data = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

static inline void tensor_ptr_array_free(TensorPtrArray* arr) {
    if (arr->data != NULL) {
        free(arr->data);
        arr->data = NULL;
    }
    arr->count = 0;
    arr->capacity = 0;
}

static inline int tensor_ptr_array_push(TensorPtrArray* arr, Tensor* t) {
    if (t == NULL) return 1;
    if (arr->count == arr->capacity) {
        int new_capacity = (arr->capacity == 0) ? 64 : arr->capacity * 2;
        Tensor** new_data = (Tensor**)realloc(arr->data, new_capacity * sizeof(Tensor*));
        if (new_data == NULL) {
            printf("tensor_ptr_array_push: ERROR: realloc failed\n");
            return 0;
        }
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->count++] = t;
    return 1;
}

static inline void tensor_set_inputs(Tensor* t, Tensor** inputs, int count) {
    if (t->_inputs != NULL) {
        free(t->_inputs);
        t->_inputs = NULL;
    }
    t->_num_inputs = 0;
    if (count <= 0 || inputs == NULL) {
        return;
    }
    t->_inputs = (Tensor**)malloc(count * sizeof(Tensor*));
    if (t->_inputs == NULL) {
        printf("tensor_set_inputs: ERROR: malloc failed\n");
        return;
    }
    for (int i = 0; i < count; ++i) {
        t->_inputs[i] = inputs[i];
    }
    t->_num_inputs = count;
}

static inline void tensor_set_inputs1(Tensor* t, Tensor* a) {
    Tensor* arr[1];
    arr[0] = a;
    tensor_set_inputs(t, arr, 1);
}

static inline void tensor_set_inputs2(Tensor* t, Tensor* a, Tensor* b) {
    Tensor* arr[2];
    arr[0] = a;
    arr[1] = b;
    tensor_set_inputs(t, arr, 2);
}

/*
  Helper: compute number of elements = shape[0] * shape[1] * ... * shape[ndim-1]
*/
static inline int tensor_numel(const Tensor* t) {
    if (t == NULL || t->ndim <= 0) {
        return 0;
    }
    int n = 1;
    for (int i = 0; i < t->ndim; ++i) {
        if (t->shape[i] == 0) {
            return 0;
        }
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
    t->grad = (float*)malloc(n * sizeof(float));
    if (t->data == NULL || t->grad == NULL) {
        printf("tensor_init: ERROR: malloc failed for %d elements\n", n);
    }
    
    // Initialize grad to zero
    memset(t->grad, 0, n * sizeof(float));

    t->_ctx = NULL;
    t->_backward = NULL;
    t->_inputs = NULL;
    t->_num_inputs = 0;
    t->visited = 0;
    t->dist_type = TENSOR_DIST_REPLICATED;
    t->dist_owner = -1;
}

/*
  Free the memory owned by this tensor.
*/
static inline void tensor_free(Tensor* t) {
    if (t->data != NULL) {
        free(t->data);
        t->data = NULL;
    }
    if (t->grad != NULL) {
        free(t->grad);
        t->grad = NULL;
    }
    t->ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
        t->shape[i] = 0;
    }
    if (t->_inputs != NULL) {
        free(t->_inputs);
        t->_inputs = NULL;
    }
    t->_num_inputs = 0;
    t->_ctx = NULL;
    t->_backward = NULL;
    t->dist_type = TENSOR_DIST_REPLICATED;
    t->dist_owner = -1;
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

static inline void tensor_set_seed(unsigned int seed) {
    srand(seed);
}

static inline float tensor_rand_uniform() {
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float tensor_randn(float mean, float stddev) {
    float u1 = tensor_rand_uniform();
    float u2 = tensor_rand_uniform();
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    float z = r * cosf(theta);
    return mean + z * stddev;
}

static inline void tensor_fill_randn(Tensor* t, float mean, float stddev) {
    int n = tensor_numel(t);
    for (int i = 0; i < n; ++i) {
        t->data[i] = tensor_randn(mean, stddev);
    }
}

/*
  Zero out the tensor data.
*/
static inline void tensor_zero(Tensor* t) {
    int n = tensor_numel(t);
    memset(t->data, 0, n * sizeof(float));
}


/*
  Zero out the tensor gradients.
*/
static inline void tensor_zero_grad(Tensor* t) {
    int n = tensor_numel(t);
    memset(t->grad, 0, n * sizeof(float));
}

/*
  Scale the tensor by a scalar value.
*/
static inline void tensor_scale(Tensor* t, float scale) {
    int n = tensor_numel(t);
    for (int i = 0; i < n; i++) {
        t->data[i] *= scale;
    }
}

/*
  Copy data from one tensor to another.
*/
static inline void tensor_copy(Tensor* dst, const Tensor* src) {
    if (dst->ndim != src->ndim) {
        printf("tensor_copy: ERROR: ndim mismatch\n");
        return;
    }
    for (int i = 0; i < dst->ndim; i++) {
        if (dst->shape[i] != src->shape[i]) {
            printf("tensor_copy: ERROR: shape mismatch\n");
            return;
        }
    }
    int n = tensor_numel(src);
    memcpy(dst->data, src->data, n * sizeof(float));
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
