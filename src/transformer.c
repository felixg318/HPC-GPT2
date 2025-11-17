#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void save_embedded_vectors (float *arr, int x, int y); //make new c file for print matrices?

//compute max score for softmax 
//computation numbers are overflowing
void simple_soft_attention(float *embedding, int batch_seq, int seq_length, int embedding_dim) {
	float *attention_scores = (float*)malloc(batch_seq * seq_length * seq_length  * sizeof(float));
	float *attention_weights = (float*)malloc(batch_seq * seq_length * seq_length  * sizeof(float)); 
	float *weighted_sums = (float*)malloc(batch_seq * seq_length * embedding_dim  * sizeof(float)); 
	
	for (int b = 0; b < batch_seq; ++b) {  
		for (int i = 0; i < seq_length; ++i) {
			for (int j = 0; j < seq_length; ++j) { 
				float dot_prod = 0.0f;
				for (int d = 0; d < embedding_dim; ++d) {
					int query_idx = b * seq_length + i * embedding_dim + d;
					int key_idx =   b * seq_length + j * embedding_dim + d;

					dot_prod += embedding[query_idx] * embedding[key_idx];
				}
				attention_scores[b * seq_length * seq_length + i * seq_length + j] = dot_prod;
			}
			//softmax normalization
			//max value is needed for numerical stability to prevent overflow in the exponential function
			float max_score = attention_scores[b * seq_length * seq_length + i * seq_length];
			for (int j = 1; j < seq_length; ++j) {
			    int idx = b * seq_length * seq_length + i * seq_length + j;
			    if (attention_scores[idx] > max_score) {
				max_score = attention_scores[idx];
			    }
			}

			float score_sum_exp = 0.0f;

			for (int j = 0; j < seq_length; ++j) {
				int idx = (b * seq_length + i) * seq_length + j;
				float tmp = exp(attention_scores[idx] - max_score);

				attention_weights[idx] = tmp;	
				score_sum_exp += tmp;
			}
			
			for (int j = 0; j < seq_length; ++j) {
				int idx = (b * seq_length + i) * seq_length + j;
				attention_weights[idx] /= score_sum_exp;	
			}
			//eof softmax
			
			//attention weights * input embeddings
			//32*128*128  x 32 * 128 * 64 = 32 * 128 * 64 matrix
			
			for (int d = 0; d < embedding_dim; ++d) {
				float weighted_sum = 0.0f;
				for(int j = 0; j < seq_length; ++j) {
					int weight_idx = b * seq_length * seq_length    + i * seq_length    + j;  
					int embed_idx =  b * seq_length * embedding_dim + j * embedding_dim + d;
					weighted_sum += attention_weights[weight_idx] * embedding[embed_idx];
				}
				weighted_sums[b * seq_length * embedding_dim + i * embedding_dim + d] = weighted_sum;
			}
		}
       	}

//	save_embedded_vectors(weighted_sums, embedding_dim, batch_seq * seq_length);
	
	free(attention_scores);
	free(attention_weights);
	free(weighted_sums);
}


