// checkpoint.h
// Minimal utilities to save/load model parameters.

#pragma once

#include <stdio.h>
#include <stdint.h>
#include "tensor.h"

static inline int save_weights(const char* path, const TensorPtrArray* params) {
    if (params == NULL || params->count <= 0) return 0;
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        printf("save_weights: failed to open %s\n", path);
        return 0;
    }
    int32_t count = params->count;
    if (fwrite(&count, sizeof(int32_t), 1, f) != 1) {
        printf("save_weights: failed to write count\n");
        fclose(f);
        return 0;
    }
    for (int i = 0; i < params->count; ++i) {
        Tensor* t = params->data[i];
        int32_t ndim = t->ndim;
        if (fwrite(&ndim, sizeof(int32_t), 1, f) != 1) {
            printf("save_weights: failed to write ndim for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        if (fwrite(t->shape, sizeof(int32_t), ndim, f) != (size_t)ndim) {
            printf("save_weights: failed to write shape for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        int32_t numel = tensor_numel(t);
        if (fwrite(&numel, sizeof(int32_t), 1, f) != 1) {
            printf("save_weights: failed to write numel for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        if (fwrite(t->data, sizeof(float), numel, f) != (size_t)numel) {
            printf("save_weights: failed to write data for tensor %d\n", i);
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return 1;
}

static inline int load_weights(const char* path, const TensorPtrArray* params) {
    if (params == NULL || params->count <= 0) return 0;
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        printf("load_weights: failed to open %s\n", path);
        return 0;
    }
    int32_t count = 0;
    if (fread(&count, sizeof(int32_t), 1, f) != 1) {
        printf("load_weights: failed to read count\n");
        fclose(f);
        return 0;
    }
    if (count != params->count) {
        printf("load_weights: tensor count mismatch: file=%d, expected=%d\n", count, params->count);
        fclose(f);
        return 0;
    }
    for (int i = 0; i < params->count; ++i) {
        Tensor* t = params->data[i];
        int32_t ndim = 0;
        if (fread(&ndim, sizeof(int32_t), 1, f) != 1) {
            printf("load_weights: failed to read ndim for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        int32_t shape_file[TENSOR_MAX_DIMS] = {0};
        if (ndim > TENSOR_MAX_DIMS) {
            printf("load_weights: ndim too large (%d) for tensor %d\n", ndim, i);
            fclose(f);
            return 0;
        }
        if (fread(shape_file, sizeof(int32_t), ndim, f) != (size_t)ndim) {
            printf("load_weights: failed to read shape for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        int32_t numel_file = 0;
        if (fread(&numel_file, sizeof(int32_t), 1, f) != 1) {
            printf("load_weights: failed to read numel for tensor %d\n", i);
            fclose(f);
            return 0;
        }
        int numel_expected = tensor_numel(t);
        if (ndim != t->ndim) {
            printf("load_weights: ndim mismatch for tensor %d (file=%d, expected=%d)\n", i, ndim, t->ndim);
            fclose(f);
            return 0;
        }
        for (int d = 0; d < ndim; ++d) {
            if (shape_file[d] != t->shape[d]) {
                printf("load_weights: shape mismatch for tensor %d at dim %d (file=%d, expected=%d)\n",
                       i, d, shape_file[d], t->shape[d]);
                fclose(f);
                return 0;
            }
        }
        if (numel_file != numel_expected) {
            printf("load_weights: numel mismatch for tensor %d (file=%d, expected=%d)\n",
                   i, numel_file, numel_expected);
            fclose(f);
            return 0;
        }
        if (fread(t->data, sizeof(float), numel_file, f) != (size_t)numel_file) {
            printf("load_weights: failed to read data for tensor %d\n", i);
            fclose(f);
            return 0;
        }
    }
    fclose(f);
    return 1;
}
