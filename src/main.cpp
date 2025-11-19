/*
 *	MPI Parallel Version
 *	Distributes batches across MPI processes for parallel execution
 */
/*
 *	MPI Parallel Version
 *	Distributes batches across MPI processes for parallel execution
 */

#include <mpi.h>
#include <iostream>
#include "tokenizor.hpp"

const int STRIDE = 1;
const int NUM_SEQ = 32;
const int SEQ_LENGTH = 128;
const int EMBEDDING_DIM = 64;

extern "C" {
	void create_batches(int *src, int *in_batch, int *target_batch, int num_seq, int seq_len, int stride, int data_size, int pad_id);
	void create_embedding (float *weights, int size, float min, float max); 
	void embed_tokens (int *input, float *weights, float *out, int batch_size, int embedding_dim);
	void inplace_add_positional(float *embedding, float *pos_weights, int batch_size, int seq_len, int embedding_dim);
	void print_array (int *a, int size);
	void print_farray (float *a, int size);
	void save_to_file(float *mat, int dim1, int dim2, int dim3, const char *filename);
	void save_4d_to_file(float *tensor, int dim1, int dim2, int dim3, int dim4, const char *filename);
	void self_attention_v1(float *embedding, float *context_vec, float *w_q, float *w_k, float *w_v, int num_seq, int seq_len, int embedding_dim, int in_dim, int out_dim, int is_training);
}

int main(int argc, char* argv[]) {
	int n_proc, rank;
	int batch_size = NUM_SEQ * SEQ_LENGTH;
	int embedding_dim = EMBEDDING_DIM;
	int in_dim = EMBEDDING_DIM;
	int out_dim = EMBEDDING_DIM;

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

	// Initialize MPI
	MPI_Comm world = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(world, &n_proc);
	MPI_Comm_rank(world, &rank);

	// Tokenization (only rank 0 does this)
	Tokenizor tokenizor;
	int data_size = 0;
	int batches = 0;
	int pad_id = 0;
	int vocab_size = 0;
	
	if (rank == 0) {
		tokenizor.set_file(argv[1]);
		tokenizor.extract_tokens();
		tokenizor.encode();
		
		data_size = tokenizor.data.size();
		batches = data_size / batch_size + 1;
		
		// Padding if needed
		if (static_cast<int>(tokenizor.data.size()) != batches * batch_size) {
			std::cout << "Need to pad input text\n";
			tokenizor.pad_data(batches * batch_size);
			data_size = tokenizor.data.size();	
			pad_id = tokenizor.get_pad_id();
		}
		
		vocab_size = tokenizor.vocab_size;
	}
	
	// Broadcast metadata to all processes
	MPI_Bcast(&data_size, 1, MPI_INT, 0, world);
	MPI_Bcast(&batches, 1, MPI_INT, 0, world);
	MPI_Bcast(&pad_id, 1, MPI_INT, 0, world);
	MPI_Bcast(&vocab_size, 1, MPI_INT, 0, world);
	int token_weights_size = vocab_size * embedding_dim;
	int pos_weights_size = SEQ_LENGTH * embedding_dim;
	int qkv_size = in_dim * out_dim;

	// Calculate work distribution for this process
	int batches_per_proc = batches / n_proc;
	int my_start_batch = rank * batches_per_proc;
	int my_end_batch = (rank == n_proc - 1) ? batches : (rank + 1) * batches_per_proc;
	int my_num_batches = my_end_batch - my_start_batch;
	
	if (rank == 0) {
		std::cout << batches << " batches of " << NUM_SEQ << " x " << SEQ_LENGTH 
		          << " tokens distributed across " << n_proc << " processes\n";
	}

	// Allocate local buffers for this process's work
	int my_batch_size = my_num_batches * batch_size;
	int *my_input_batch = new int[my_batch_size];
	int *my_target_batch = new int[my_batch_size];
	float *my_embedding_batch = new float[my_batch_size * embedding_dim];
	float *my_context_vector = new float[my_batch_size * embedding_dim];

	// Allocate shared weights (all processes need these)
	float *token_weights = new float[token_weights_size]; 
	float *pos_weights = new float[pos_weights_size];
	float *W_Q = new float[qkv_size];
	float *W_K = new float[qkv_size];
	float *W_V = new float[qkv_size];
	
	// Initialize embeddings on rank 0, then broadcast to all
	if (rank == 0) {
		create_embedding(token_weights, token_weights_size, -5.0, 5.0);
		create_embedding(pos_weights, pos_weights_size, -0.05, 0.05);
		create_embedding(W_Q, qkv_size, -0.05, 0.05);
		create_embedding(W_K, qkv_size, -0.05, 0.05);
		create_embedding(W_V, qkv_size, -0.05, 0.05);
	}

	// Broadcast shared weights to all processes
	MPI_Bcast(token_weights, token_weights_size, MPI_FLOAT, 0, world);
	MPI_Bcast(pos_weights, pos_weights_size, MPI_FLOAT, 0, world);
	MPI_Bcast(W_Q, qkv_size, MPI_FLOAT, 0, world);
	MPI_Bcast(W_K, qkv_size, MPI_FLOAT, 0, world);
	MPI_Bcast(W_V, qkv_size, MPI_FLOAT, 0, world);

	// Broadcast tokenized data to all processes
	int *tokenized_data = nullptr;
	if (rank == 0) {
		tokenized_data = tokenizor.data.data();
	} else {
		tokenized_data = new int[data_size];
	}
	MPI_Bcast(tokenized_data, data_size, MPI_INT, 0, world);

	int is_training = 1;

	// Each process handles only its assigned batches
	for (int b = my_start_batch; b < my_end_batch; b++) {
		int local_b = b - my_start_batch;  // Local batch index for this process
		int batch_offset = local_b * batch_size;
		int embedding_offset = batch_offset * embedding_dim;
		int global_batch_offset = b * batch_size;  // Offset in global data
		
		create_batches(tokenized_data + global_batch_offset, 
		               my_input_batch + batch_offset, 
		               my_target_batch + batch_offset, 
		               NUM_SEQ, SEQ_LENGTH, STRIDE, data_size, pad_id);
		
		embed_tokens(my_input_batch + batch_offset, 
		             token_weights, 
		             my_embedding_batch + embedding_offset, 
		             batch_size, embedding_dim);
		
		inplace_add_positional(my_embedding_batch + embedding_offset, 
		                       pos_weights, batch_size, SEQ_LENGTH, embedding_dim);
		
		self_attention_v1(my_embedding_batch + embedding_offset, 
		                  my_context_vector + embedding_offset,  
		                  W_Q, W_K, W_V, NUM_SEQ, SEQ_LENGTH, 
		                  embedding_dim, in_dim, out_dim, is_training);
	}

	is_training = 0;

	// Gather results back to rank 0
	float *context_vector = nullptr;
	if (rank == 0) {
		context_vector = new float[batches * batch_size * embedding_dim];
	}

	// Calculate send counts and displacements for MPI_Gatherv (handles uneven distribution)
	int *recvcounts = nullptr;
	int *displs = nullptr;
	
	if (rank == 0) {
		recvcounts = new int[n_proc];
		displs = new int[n_proc];
		
		for (int i = 0; i < n_proc; i++) {
			int start = i * batches_per_proc;
			int end = (i == n_proc - 1) ? batches : (i + 1) * batches_per_proc;
			int num_batches = end - start;
			recvcounts[i] = num_batches * batch_size * embedding_dim;
			displs[i] = start * batch_size * embedding_dim;
		}
	}

	MPI_Gatherv(my_context_vector, my_batch_size * embedding_dim, MPI_FLOAT,
	            context_vector, recvcounts, displs, MPI_FLOAT, 0, world);

	// Save results (only rank 0)
	if (rank == 0) {
		save_4d_to_file(context_vector, NUM_SEQ, SEQ_LENGTH, embedding_dim, batches, "results.txt");
		std::cout << "Results saved to results.txt\n";
	}

	// Cleanup
	delete[] my_input_batch;
	delete[] my_target_batch;
	delete[] my_embedding_batch;
	delete[] my_context_vector;

	delete[] token_weights;
	delete[] pos_weights;
	delete[] W_Q;
	delete[] W_K;
	delete[] W_V;

	if (rank != 0) {
		delete[] tokenized_data;
	}

	if (rank == 0) {
		delete[] context_vector;
		delete[] recvcounts;
		delete[] displs;
	}

	MPI_Finalize();
	return 0;
}
