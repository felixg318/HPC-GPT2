#include<stdio.h>

void print_array(int *a, int size) {
        for (int i = 0; i < size; ++i)
                printf("%d ", a[i]);
        printf("\n");
}
void print_farray(float *a, int size) {
        for (int i = 0; i < size; ++i)
                printf("%f ", a[i]);
        printf("\n");
}

void save_to_file(float *mat, int dim1, int dim2, int dim3, const char *filename) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                fprintf(f, "%.6f ", mat[i * dim2 * dim3 + j * dim3 + k]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void save_2d_to_file(float *mat, int rows, int cols, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(f, "%.6f", mat[i * cols + j]);
            if (j < cols - 1) fprintf(f, " ");  // Space between columns
        }
        fprintf(f, "\n");  // Newline after each row
    }

    fclose(f);
}

void save_4d_to_file(float *tensor, int dim1, int dim2, int dim3, int dim4, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }
    
    // Write dimensions as header
    fprintf(f, "# Shape: [%d, %d, %d, %d]\n", dim1, dim2, dim3, dim4);
    
    for (int i = 0; i < dim1; ++i) {
        fprintf(f, "# Batch %d\n", i);
        for (int j = 0; j < dim2; ++j) {
            fprintf(f, "## Sequence %d\n", j);
            for (int k = 0; k < dim3; ++k) {
                for (int l = 0; l < dim4; ++l) {
                    int idx = i * dim2 * dim3 * dim4 + 
                             j * dim3 * dim4 + 
                             k * dim4 + 
                             l;
                    fprintf(f, "%.6f", tensor[idx]);
                    if (l < dim4 - 1) fprintf(f, " ");
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    
    fclose(f);
    printf("Saved 4D tensor to %s\n", filename);
}
