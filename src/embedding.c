#include <stdlib.h>
#include <stdio.h>

const int STRIDE = 1;
const int BATCH_SEQ = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

float random_float(float min, float max) {
	return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

void create_embedding(float *weights, int size, float min, float max) {
	for (int i = 0; i < size; ++i) 
		weights[i] = random_float(min, max);
}

void create_batches(int *src, int *in_batch, int *target_batch, int batch_seq, int seq_len, int stride, int src_size, int pad_id) {
	for (int i = 0; i < batch_seq; ++i) {
		for (int j = 0; j < seq_len; ++j) {
			int idx = i * seq_len + j;

			in_batch[idx] = (idx < src_size) ? src[idx] : pad_id;
			target_batch[idx] = (idx + STRIDE < src_size) ? src[idx + STRIDE] : pad_id;
		}
	}
}

//can cause bounds checking error
void embed_tokens(int *input, float *weights, float *out, int input_size, int embedding_dim) {
	for (int token = 0; token < input_size; ++token) 
		for (int j = 0; j < embedding_dim; ++j) 
			out[token * embedding_dim + j] = weights[input[token] * embedding_dim + j];
}

void inplace_add_positional(float *embedding, float *pos_weights, int batch_size, int seq_len, int embedding_dim) {
	for (int i = 0; i < batch_size; ++i) { 
		int pos = i % seq_len;
		for (int j = 0; j < embedding_dim; ++j) {
			embedding[i * embedding_dim + j] += pos_weights[pos * embedding_dim + j];
		}
	}
}

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
