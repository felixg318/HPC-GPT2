// checkpoint.h
// Minimal utilities to save/load model parameters.

#pragma once

#include <stdio.h>
#include <stdint.h>
#include "tensor.h"
#include <mpi.h>

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

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int ok = 1;
    FILE* f = NULL;
    if (rank == 0) {
        f = fopen(path, "rb");
        if (f == NULL) {
            printf("load_weights: failed to open %s\n", path);
            ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        if (rank == 0 && f != NULL) fclose(f);
        return 0;
    }

    int32_t count = 0;
    if (rank == 0) {
        if (fread(&count, sizeof(int32_t), 1, f) != 1) {
            printf("load_weights: failed to read count\n");
            ok = 0;
        }
    }
    MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        if (rank == 0 && f != NULL) fclose(f);
        return 0;
    }

    if (count != params->count) {
        if (rank == 0) {
            printf("load_weights: tensor count mismatch: file=%d, expected=%d\n", count, params->count);
        }
        ok = 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!ok) {
        return 0;
    }

    for (int i = 0; i < params->count; ++i) {
        Tensor* t = params->data[i];
        int32_t ndim_file = 0;
        int32_t shape_file[TENSOR_MAX_DIMS] = {0};
        int32_t numel_file = 0;
        float* file_tensor = NULL;

        if (rank == 0) {
            if (fread(&ndim_file, sizeof(int32_t), 1, f) != 1) {
                printf("load_weights: failed to read ndim for tensor %d\n", i);
                ok = 0;
            }
        }
        MPI_Bcast(&ndim_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) break;

        if (ndim_file > TENSOR_MAX_DIMS) {
            if (rank == 0) printf("load_weights: ndim too large (%d) for tensor %d\n", ndim_file, i);
            ok = 0;
        }
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) break;

        if (rank == 0) {
            if (fread(shape_file, sizeof(int32_t), ndim_file, f) != (size_t)ndim_file) {
                printf("load_weights: failed to read shape for tensor %d\n", i);
                ok = 0;
            }
        }
        MPI_Bcast(shape_file, TENSOR_MAX_DIMS, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) break;
        
        if (rank == 0) {
            if (fread(&numel_file, sizeof(int32_t), 1, f) != 1) {
                printf("load_weights: failed to read numel for tensor %d\n", i);
                ok = 0;
            }
        }
        MPI_Bcast(&numel_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!ok) break;
        
        file_tensor = (float*)malloc(numel_file * sizeof(float));
        if (file_tensor == NULL) {
            if (rank == 0) printf("load_weights: failed to allocate buffer for tensor %d\n", i);
            ok = 0;
        } else if (rank == 0) {
            if (fread(file_tensor, sizeof(float), numel_file, f) != (size_t)numel_file) {
                 printf("load_weights: failed to read data for tensor %d\n", i);
                 ok = 0;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) {
            free(file_tensor);
            break;
        }
        
        MPI_Bcast(file_tensor, numel_file, MPI_FLOAT, 0, MPI_COMM_WORLD);

        int shape_mismatch = 0;
        if (t->ndim != ndim_file) {
            shape_mismatch = 1;
        } else {
            for (int d = 0; d < ndim_file; ++d) {
                if (shape_file[d] != t->shape[d]) {
                    shape_mismatch = 1;
                    break;
                }
            }
        }

        if (shape_mismatch) {
            if (t->ndim == 2 && ndim_file == 2 && shape_file[0] == t->shape[0] * world_size && shape_file[1] == t->shape[1]) {
                // Row-parallel
                int rows_per_rank = t->shape[0];
                int row_offset = rank * rows_per_rank;
                int cols = t->shape[1];
                memcpy(t->data, file_tensor + row_offset * cols, rows_per_rank * cols * sizeof(float));
            } else if (t->ndim == 2 && ndim_file == 2 && shape_file[0] == t->shape[0] && shape_file[1] == t->shape[1] * world_size) {
                // Col-parallel
                int cols_per_rank = t->shape[1];
                int col_offset = rank * cols_per_rank;
                int rows = t->shape[0];
                for (int r = 0; r < rows; r++) {
                    memcpy(t->data + r * cols_per_rank, file_tensor + r * shape_file[1] + col_offset, cols_per_rank * sizeof(float));
                }
            } else if (t->ndim == 1 && ndim_file == 1 && shape_file[0] == t->shape[0] * world_size) {
                // 1D tensor, partitioned
                int elements_per_rank = t->shape[0];
                int offset = rank * elements_per_rank;
                memcpy(t->data, file_tensor + offset, elements_per_rank * sizeof(float));
            } else {
                 if(rank == 0) printf("load_weights: unhandled partitioned tensor %d\n", i);
                 ok = 0;
            }
        } else {
            memcpy(t->data, file_tensor, numel_file * sizeof(float));
        }
        free(file_tensor);
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) break;
    }
    
    if (rank == 0 && f != NULL) {
        fclose(f);
    }
    return ok;
}
