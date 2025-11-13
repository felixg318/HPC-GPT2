#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

void create_batches (int *src, int *in_batch, int *target_batch, int src_size, int batch_size, int *batch_idx);
void create_embeddings (float *weights, int size, float min, float max); 
void embed_tokens (int *input, float *weights, float *out, int input_size, int vocab_size, int embedding_dim);
void mat_inplace_add (float *A, float *B, int rows, int cols);
void print_array (int *a, int size);
void print_farray (float *a, int size);
void save_embedded_vectors (float *arr, int x, int y);

int main(int argc, char** argv) {
	Tokenizor tokenizor;
	int *input_batch, *target_batch;
	float *embedding_weight_layer, *embedded_tokens, *pos_data;
	int batch_size = BATCH_SIZE * SEQ_LENGTH;
	int embedding_dim = EMBEDDING_DIM;
	int batches, batch_idx = 0; //keep track of index to prevent garbage values
	int vocab_size, token_size, embedding_size, embedded_tokens_size;

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

	tokenizor.set_file(argv[1]);
	tokenizor.extract_tokens();
	tokenizor.encode();

	vocab_size = tokenizor.vocab_size;
	token_size = tokenizor.data.size();

	batches = token_size / batch_size;
	embedding_size = vocab_size * embedding_dim;
	embedded_tokens_size = token_size * embedding_dim;
	
	input_batch = new int[batch_size];
	target_batch = new int[batch_size];
	
	embedding_weight_layer = new float[embedding_size]; //each token has embedding_dim vector (embedding_dim is also offset per token
	
	embedded_tokens = new float[embedded_tokens_size];
	pos_data = new float[embedded_tokens_size];

	std::cout << "Need " << batches << " batches of " << BATCH_SIZE << " x " << SEQ_LENGTH << " tokens\n";

	
	create_embeddings(embedding_weight_layer, embedding_size, -5.0, 5.0);
	create_embeddings(pos_data, embedded_tokens_size, -0.05, 0.05);

//	Batching is going to be a huge pain in the ass when we implement MPI since
//	technically embedding tokens is to be done in batches
//	create_batches(tokenizor.data.data(), input_batch, target_batch, token_size, batch_size, &batch_idx);
//	std::cout << batch_idx << '\n';

	embed_tokens(tokenizor.data.data(), embedding_weight_layer, embedded_tokens, token_size, vocab_size, embedding_dim);

	mat_inplace_add(embedded_tokens, pos_data, token_size, embedding_dim); //safe to assume same dimensions 

//	save_embedded_vectors(embedded_tokens, embedding_dim, token_size);

	tokenizor.decode(); //fix decode where we send a vector of token ids in the parameter
	
	delete[] input_batch;
	delete[] target_batch;
	delete[] embedding_weight_layer;
	delete[] pos_data;
	delete[] embedded_tokens;

	return 0;
}
