#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

void embed_tokens(int *input, float *weights, float *out, int input_size, int vocab_size, int embedding_dim);
void create_embedding_layer(float *weights, int size); 
void create_batches(int *src, int *in_batch, int *target_batch, int src_size, int batch_size, int &batch_idx);
void print_array (int *a, int size);
void print_farray (float *a, int size);
void save_embedded_vectors(float *arr, int x, int y);

int main(int argc, char** argv) {
	Tokenizor tokenizor;
	int *input_batch, *target_batch;
	float *embedding_weights, *embedded_tokens;
	int batch_size = BATCH_SIZE * SEQ_LENGTH;
	int embedding_dim = EMBEDDING_DIM;
	int batches, batch_idx = 0; //keep track of index to prevent garbage values
	int vocab_size, embedding_size, embedded_tokens_size;

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

	tokenizor.set_file(argv[1]);
	tokenizor.extract_tokens();
	tokenizor.encode();

	vocab_size = tokenizor.vocab_size;
	batches = tokenizor.data.size() / batch_size;
	embedding_size = vocab_size * embedding_dim;
	embedded_tokens_size = tokenizor.data.size() * embedding_dim;
	
	input_batch = new int[batch_size];
	target_batch = new int[batch_size];
	embedding_weights = new float[embedding_size]; //each token has embedding_dim vector (embedding_dim is also offset per token)
	embedded_tokens = new float[embedded_tokens_size];
	std::cout << "Need " << batches << " batches of " << BATCH_SIZE << " x " << SEQ_LENGTH << " tokens\n";

	
	create_embedding_layer(embedding_weights, embedding_size);
//	
//	Batching is going to be a huge pain in the ass when we implement MPI since
//	technically embedding tokens is to be done in batches
//	create_batches(tokenizor.data.data(), input_batch, target_batch, tokenizor.data.size(), batch_size, batch_idx);
//

	embed_tokens(tokenizor.data.data(), embedding_weights, embedded_tokens, tokenizor.data.size(), vocab_size, embedding_dim);
	save_embedded_vectors(embedded_tokens, embedding_dim, tokenizor.data.size());

//	encode_pos_data();

	delete[] input_batch;
	delete[] target_batch;
	delete[] embedding_weights;
	delete[] embedded_tokens;


	tokenizor.decode(); //fix decode where we send a vector of token ids in the parameter

	return 0;
}
