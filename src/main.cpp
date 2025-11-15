#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int BATCH_SEQ = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

void create_batches (int *src, int *in_batch, int *target_batch, int size, int pad_id);
void create_embedding (float *weights, int size, float min, float max); 
void embed_tokens (int *input, float *weights, float *out, int batch_size, int embedding_dim);
void inplace_add_positional(float *embedding, int *tokens, float *pos_weights, int batch_size, int embedding_dim);
void mat_inplace_add (float *A, float *B, int rows, int cols);
void print_array (int *a, int size);
void print_farray (float *a, int size);
void save_embedded_vectors (float *arr, int x, int y);

int main(int argc, char** argv) {
	Tokenizor tokenizor;
	int *input_batch, *target_batch;
	float *token_weights, *embedding_batch, *pos_weights;
	int batch_size = BATCH_SEQ * SEQ_LENGTH;
	int embedding_dim = EMBEDDING_DIM;
	int batches; 
	int vocab_size, data_size, embedding_size, token_weights_size;
	int pad_id; 
	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

	tokenizor.set_file(argv[1]);
	tokenizor.extract_tokens();
	tokenizor.encode();
	
	data_size = tokenizor.data.size();
	batches = data_size / batch_size + 1;
	
	if (tokenizor.data.size() != batches * batch_size) {
		std::cout << "Need to pad input text\n";
		tokenizor.pad_data(batches * batch_size);
		
		data_size = tokenizor.data.size();	
		pad_id = tokenizor.get_pad_id(); //important for batches
	}
	
	vocab_size = tokenizor.vocab_size;

	token_weights_size = vocab_size * embedding_dim;
	embedding_size = data_size * embedding_dim;
	
	input_batch = new int[batch_size]; 
	target_batch = new int[batch_size];
	embedding_batch = new float[batch_size * embedding_dim];
	
	token_weights = new float[token_weights_size]; 
	pos_weights = new float[embedding_size];

	std::cout<< batches << " batches of " << BATCH_SEQ << " x " << SEQ_LENGTH << " tokens\n";
	
	create_embedding(token_weights, token_weights_size, -5.0, 5.0);
	create_embedding(pos_weights, embedding_size, -0.05, 0.05);

	//potentially buggy
	//add bounds checking
	//to add to gpu, load weights and positional data to memory for cuda kernels
	for (int b = 0; b < batches; b++) {
		int batch_offset = b * batch_size;
		
		create_batches(tokenizor.data.data() + batch_offset, input_batch, target_batch, batch_size, pad_id);
		
		embed_tokens(input_batch, token_weights, embedding_batch, batch_size, embedding_dim);
		inplace_add_positional(embedding_batch, input_batch, pos_weights, batch_size, embedding_dim);
	
	}


//	For use later:
//	save_embedded_vectors(embedded_tokens, embedding_dim, data_size);
//	tokenizor.decode(array, size);

	delete[] input_batch;
	delete[] target_batch;
	delete[] embedding_batch;

	delete[] token_weights;
	delete[] pos_weights;

	return 0;
}
