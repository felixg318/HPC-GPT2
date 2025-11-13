#include <stdlib.h>
#include <stdio.h>

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

float random_float(float min, float max) {
	return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

void create_embedding_layer(float *weights, int size) {
	for (int i = 0; i < size; ++i) 
		weights[i] = random_float(-5.0, 5.0);
}

void create_batches(int *src, int *in_batch, int *target_batch, int src_size, int batch_size, int &batch_idx) {
	for (int i = 0; i < batch_size; ++i) {
		in_batch[i] = (batch_idx < src_size) ? src[i] : 0;
		target_batch[i] = (batch_idx < src_size) ? src[i + STRIDE] : 0;
		batch_idx++;
	}
}

void embed_tokens(int *input, float *weights, float *out, int input_size, int vocab_size, int embedding_dim) {
	int embedded_tokens_size = input_size * embedding_dim;

	for (int token = 0; token < input_size; ++token) 
		for (int j = 0; j < embedding_dim; ++j) 
			out[token * embedding_dim + j] = weights[input[token] * embedding_dim + j];
}

void print_array(int *a, int size) {
	for (int i = 0; i < size; ++i)
		printf("%d ", a[i]);
}
void print_farray(float *a, int size) {
	for (int i = 0; i < size; ++i)
		printf("%f ", a[i]);
}

void save_embedded_vectors(float *arr, int x, int y) {
	FILE *myfile = fopen("embedded_vector.txt", "w");

	fprintf(myfile, "[");	
	for (int i = 0; i < y; ++i) {
		fprintf(myfile, "[");	
		for (int j = 0; j < x; ++j) {
			fprintf(myfile, "%f", arr[i * x + j]);
			if (j < x - 1)
				fprintf(myfile, ", ");
		}
		fprintf(myfile, "]");
		if (i < y - 1)
			fprintf(myfile, ", \n");	
	}
	
	fprintf(myfile, "]");	

	fclose(myfile);
}
