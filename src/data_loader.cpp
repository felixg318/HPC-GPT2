#include <iostream>

const int STRIDE = 1;
const int BATCH_SIZE = 32;
const int SEQ_LENGTH = 128;

void create_batches(int *src, int *in_batch, int *target_batch, int src_size, int batch_size, int &batch_idx) {
	for (int i = 0; i < batch_size; ++i) {
	        in_batch[i] = (batch_idx < src_size) ? src[i] : 0;
        	target_batch[i] = (batch_idx < src_size) ? src[i + STRIDE] : 0;
		
		batch_idx++;
	}
}

void print_batch(int *in_batch, int *target_batch, int batch_size) {
	for (int i = 0; i < 10; ++i)
		std::cout<< in_batch[i];
	std::cout << '\n';
	for (int i = 0; i < 10; ++i)
		std::cout<< target_batch[i];
	std::cout << '\n';
}
