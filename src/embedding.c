#include <stdlib.h>
#include <stdio.h>

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

float random_float(float min, float max) {
	return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

void create_embeddings(float *pos_data, int size, float min, float max) {
	for (int i = 0; i < size; ++i) 
		pos_data[i] = random_float(min, max);
}

//IMPORTANT CHANGE 0 to a Padding token
void create_batches(int *src, int *in_batch, int *target_batch, int src_size, int batches, int pad_id) {
	int batch_size = BATCH_SIZE * SEQ_LENGTH;

	for (int b = 0; b < batches; ++b) {
		for (int i = 0; i < BATCH_SIZE; ++i) {
			for (int j = 0; j < SEQ_LENGTH; ++j) {
				int idx = b * batch_size + i * SEQ_LENGTH + j;

				in_batch[idx] = (idx < src_size) ? src[idx] : pad_id;
				target_batch[idx] = (idx + STRIDE < src_size) ? src[idx + STRIDE] : pad_id;
				
			}
		}
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
