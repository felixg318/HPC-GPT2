#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

void create_batches (int *src, int *in_batch, int *target_batch, int src_size, int batches, int pad_id);
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
	int batches; 
	int vocab_size, data_size, embedding_size, embedded_tokens_size;
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

	embedding_size = vocab_size * embedding_dim;
	embedded_tokens_size = data_size * embedding_dim;
	
	input_batch = new int[batches * batch_size]; 
	target_batch = new int[batches * batch_size];
	
	embedding_weight_layer = new float[embedding_size]; 
	
	embedded_tokens = new float[embedded_tokens_size];
	pos_data = new float[embedded_tokens_size];

	std::cout<< batches << " batches of " << BATCH_SIZE << " x " << SEQ_LENGTH << " tokens\n";

	
	create_embeddings(embedding_weight_layer, embedding_size, -5.0, 5.0);
	create_embeddings(pos_data, embedded_tokens_size, -0.05, 0.05);

//	Batching is going to be a huge pain in the ass when we implement MPI since
//	technically embedding tokens is to be done in batches

	create_batches(tokenizor.data.data(), input_batch, target_batch, data_size, batches, pad_id);
	

	embed_tokens(tokenizor.data.data(), embedding_weight_layer, embedded_tokens, data_size, vocab_size, embedding_dim);

	mat_inplace_add(embedded_tokens, pos_data, data_size, embedding_dim); //safe to assume same dimensions 

//	save_embedded_vectors(embedded_tokens, embedding_dim, data_size);

//	tokenizor.decode(array, size);
	
	delete[] input_batch;
	delete[] target_batch;
	delete[] embedding_weight_layer;
	delete[] pos_data;
	delete[] embedded_tokens;

	return 0;
}
