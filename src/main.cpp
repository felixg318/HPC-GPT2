/*
 *	Serial Verision
 *	Added comments for MPI routines for future
 *	
 */

//#include <mpi.h>

#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int NUM_SEQ = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

//we can refactor these parameters and put arrays in structs
extern "C" {
	void create_batches(int *src, int *in_batch, int *target_batch, int num_seq, int seq_len, int stride, int data_size, int pad_id);
	void create_embedding (float *weights, int size, float min, float max); 
	void embed_tokens (int *input, float *weights, float *out, int batch_size, int embedding_dim);
	void inplace_add_positional(float *embedding, float *pos_weights, int batch_size, int seq_len, int embedding_dim);
	void print_array (int *a, int size);
	void print_farray (float *a, int size);
	void save_to_file(float *mat, int dim1, int dim2, int dim3, const char *filename);
	void save_4d_to_file(float *tensor, int dim1, int dim2, int dim3, int dim4, const char *filename);
	void self_attention_v1(float *embedding, float *context_vec, float *w_q, float *w_k, float *w_v, int num_seq, int seq_len, int embedding_dim, int in_dim, int out_dim);
}

int main(int argc, char* argv[]) {
//	int n_proc, my_work, rank, n;

	Tokenizor tokenizor;
	int *input_batch, *target_batch;
	float *token_weights, *embedding_batch, *pos_weights;
	int batch_size = NUM_SEQ * SEQ_LENGTH;
	int embedding_dim = EMBEDDING_DIM;
	int in_dim = EMBEDDING_DIM;
	int out_dim = EMBEDDING_DIM;
	int batches; 
	int vocab_size, data_size, pos_weights_size, token_weights_size;
	int pad_id {};
	int qkv_size;
	float *W_Q, *W_K, *W_V;	
	float *context_vector;

//	long mpi_start, mpi_end, mpi_elapsed;
//	long host_start, host_end, host_elapsed;
//	long dev_start, dev_end, dev_elapsed;

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

//	MPI_Comm world = MPI_COMM_WORLD
//
//	MPI_Init();
//	MPI_Comm_size(world, &n_proc);
//	MPI_Comm_rank(world, &rank);
//
	tokenizor.set_file(argv[1]);
	tokenizor.extract_tokens();
	tokenizor.encode();
	
	data_size = tokenizor.data.size();
	batches = data_size / batch_size + 1;
	
	if (static_cast<int>(tokenizor.data.size()) != batches * batch_size) {
		std::cout << "Need to pad input text\n";
		tokenizor.pad_data(batches * batch_size);
		
		data_size = tokenizor.data.size();	
		pad_id = tokenizor.get_pad_id(); //important for batches
	}
	
	vocab_size = tokenizor.vocab_size;

	token_weights_size = vocab_size * embedding_dim;
	pos_weights_size = SEQ_LENGTH * embedding_dim;
	qkv_size = in_dim * out_dim;
		
	input_batch = new int[batches * batch_size];
	target_batch = new int[batches * batch_size];
	embedding_batch = new float[batches * batch_size * embedding_dim];
	
	token_weights = new float[token_weights_size]; 
	pos_weights = new float[pos_weights_size];

	context_vector = new float[batches * batch_size * embedding_dim];
	W_Q = new float[qkv_size];
	W_K = new float[qkv_size];
	W_V = new float[qkv_size];
	
	std::cout<< batches << " batches of " << NUM_SEQ << " x " << SEQ_LENGTH << " tokens\n";
	
	create_embedding(token_weights, token_weights_size, -5.0, 5.0);
	create_embedding(pos_weights, pos_weights_size, -0.05, 0.05);

	create_embedding(W_Q, qkv_size, -0.05, 0.05);
	create_embedding(W_K, qkv_size, -0.05, 0.05);
	create_embedding(W_V, qkv_size, -0.05, 0.05);

	//potentially buggy
	//add bounds checking
	//to add to gpu, load weights and positional data to memory for cuda kernels
	for (int b = 0; b < batches; b++) {
		int batch_offset = b * batch_size;
		int embedding_offset = batch_offset * embedding_dim;
		
		create_batches(tokenizor.data.data() + batch_offset, input_batch + batch_offset, target_batch + batch_offset, NUM_SEQ, SEQ_LENGTH, STRIDE, data_size, pad_id);
		embed_tokens(input_batch + batch_offset, token_weights, embedding_batch + embedding_offset, batch_size, embedding_dim);
		inplace_add_positional(embedding_batch + embedding_offset, pos_weights, batch_size, SEQ_LENGTH, embedding_dim);
		
		self_attention_v1(embedding_batch + embedding_offset, context_vector + embedding_offset,  W_Q, W_K, W_V, NUM_SEQ, SEQ_LENGTH, embedding_dim, in_dim, out_dim);
	}

//	save_4d_to_file(context_vector, NUM_SEQ, SEQ_LENGTH, embedding_dim, batches, "results.txt");
	delete[] input_batch;
	delete[] target_batch;
	delete[] embedding_batch;

	delete[] token_weights;
	delete[] pos_weights;
	delete[] W_Q;
	delete[] W_K;
	delete[] W_V;
	delete[] context_vector;

//	MPI_Finalize();
	return 0;
}
