// autograd.h
// Generic autograd engine using tensor input lists

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include "tensor.h"

static inline void build_topo(Tensor* v,
                              Tensor*** topo,
                              int* n_topo,
                              int* capacity) {
    if (v == NULL || v->visited) {
        return;
    }
    v->visited = 1;

    for (int i = 0; i < v->_num_inputs; ++i) {
        build_topo(v->_inputs[i], topo, n_topo, capacity);
    }

    if (*n_topo >= *capacity) {
        int new_capacity = (*capacity == 0) ? 1024 : (*capacity * 2);
        Tensor** new_topo = (Tensor**)realloc(*topo, new_capacity * sizeof(Tensor*));
        if (new_topo == NULL) {
            printf("build_topo: ERROR: realloc failed\n");
            return;
        }
        *topo = new_topo;
        *capacity = new_capacity;
    }
    (*topo)[(*n_topo)++] = v;
}

static inline void backward(Tensor* t) {
    if (t == NULL || t->grad == NULL) {
        printf("backward: ERROR: grad is NULL\n");
        return;
    }

    int n = tensor_numel(t);
    for (int i = 0; i < n; ++i) {
        t->grad[i] = 1.0f;
    }

    int topo_capacity = 1024;
    Tensor** topo = (Tensor**)malloc(topo_capacity * sizeof(Tensor*));
    if (topo == NULL) {
        printf("backward: ERROR: malloc failed\n");
        return;
    }
    int n_topo = 0;
    build_topo(t, &topo, &n_topo, &topo_capacity);

    for (int i = n_topo - 1; i >= 0; --i) {
        if (topo[i]->_backward != NULL) {
            topo[i]->_backward(topo[i]);
        }
        topo[i]->visited = 0;
    }

    free(topo);
}
