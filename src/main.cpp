#include <iostream>
#include "tokenizor.h"

const int STRIDE = 1;
//think of 32x128 matrix
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;

void create_batches(int *src, int *in_batch, int *target_batch, int src_size, int batch_size, int &batch_idx);
void print_batch(int *in_batch, int *target_batch, int batch_size);

int main(int argc, char** argv) {
	Tokenizor tokenizor;
	int *input_batch, *target_batch;
	int batch_size = BATCH_SIZE * SEQ_LENGTH;
	int batches, batch_idx = 0; //keep track of index to prevent garbage values

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}

	tokenizor.set_file(argv[1]);
	tokenizor.extract_tokens();

	tokenizor.encode();

	input_batch = new int[BATCH_SIZE * SEQ_LENGTH];
	target_batch = new int[BATCH_SIZE * SEQ_LENGTH];

	batches = tokenizor.data.size() / batch_size;
	std::cout << "Need " << batches << " batches of " << BATCH_SIZE << " x " << SEQ_LENGTH << " tokens\n";

	create_batches(tokenizor.data.data(), input_batch, target_batch, tokenizor.data.size(), batch_size, batch_idx);
//	print_batch(input_batch, target_batch, batch_size);
	
	std::cout << batch_idx << '\n';
	delete[] input_batch;
	delete[] target_batch;

	tokenizor.decode(); //fix decode where we send a vector of token ids in the parameter

	return 0;
}
