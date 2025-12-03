// mpi_utils.h
// Minimal MPI helpers for data-parallel training.

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "tensor.h"

// Initialize rank/world_size when MPI is available.
static inline void mpi_get_rank_world(int* rank_out, int* world_out) {
#ifdef USE_MPI
    int r = 0, w = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &w);
    if (rank_out) *rank_out = r;
    if (world_out) *world_out = w;
#else
    if (rank_out) *rank_out = 0;
    if (world_out) *world_out = 1;
#endif
}

// Broadcast all parameter tensors from root to every rank.
static inline void mpi_broadcast_parameters(Tensor** params, int count, int root_rank
#ifdef USE_MPI
                                            , MPI_Comm comm
#endif
) {
    (void)root_rank;
    (void)params;
    (void)count;
#ifdef USE_MPI
    for (int i = 0; i < count; ++i) {
        Tensor* t = params[i];
        if (t == NULL || t->data == NULL) continue;
        int n = tensor_numel(t);
        if (n <= 0) continue;
        MPI_Bcast(t->data, n, MPI_FLOAT, root_rank, comm);
    }
#endif
}

// Allreduce gradients across ranks and scale by 1/world_size (replicated params).
static inline void mpi_allreduce_grads(Tensor** params, int count, int world_size) {
    if (params == NULL || count <= 0 || world_size <= 1) return;

    int total = 0;
    for (int i = 0; i < count; ++i) {
        if (params[i]) total += tensor_numel(params[i]);
    }
    if (total <= 0) return;

    float* buffer = (float*)malloc((size_t)total * sizeof(float));
    if (buffer == NULL) {
        printf("mpi_allreduce_grads: failed to allocate buffer of %d floats\n", total);
        return;
    }

    int offset = 0;
    for (int i = 0; i < count; ++i) {
        Tensor* t = params[i];
        int n = tensor_numel(t);
        if (n <= 0) continue;
        memcpy(buffer + offset, t->grad, n * sizeof(float));
        offset += n;
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, buffer, offset, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif

    float scale = 1.0f / (float)world_size;
    offset = 0;
    for (int i = 0; i < count; ++i) {
        Tensor* t = params[i];
        int n = tensor_numel(t);
        if (n <= 0) continue;
        for (int j = 0; j < n; ++j) {
            t->grad[j] = buffer[offset + j] * scale;
        }
        offset += n;
    }
    free(buffer);
}

// Reduce a scalar loss across ranks (sum) and return average.
static inline float mpi_allreduce_loss(float local_loss) {
    float total = local_loss;
    int world = 1;
    mpi_get_rank_world(NULL, &world);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (world > 0) total /= (float)world;
    return total;
}
