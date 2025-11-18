#include <math.h>
#include <stdio.h>
#include <stdlib.h>
/*
 *
 *	THIS FILE IS A WORK IN PROGRESS!!!
 *
 *
 * */


void save_embedded_vectors (float *arr, int x, int y); //make new c file for print matrices?
void create_embedding(float *w, int size, float min, float max);

//compute max score for softmax 
//computation numbers are overflowing
//to be further improved
/*
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
*/

/* 
 * q,k,v weight matrices are parameters
 * context vector is a parameter
 *
 * change this when finished
 *
 * */
void self_attention_v1(float *embedding, int batch_seq, int seq_len, int embedding_dim, int in_dim, int out_dim) {
	float *Q, *K, *V;
	float *W_Q, *W_K, *W_V;

	float *attention_scores;
	float *attention_weights;
	float *weighted_sums;
	
	int size = in_dim * out_dim;

	if (embedding_dim != in_dim) {
		printf("Bad dimensions for self_attention_v1. Embedding dimension does not match in_dim for weight matrices\n");
		return;
	}

	//to do outside of this function when done
	W_Q = (float*)malloc(size * sizeof(float));
	W_K = (float*)malloc(size * sizeof(float));
	W_V = (float*)malloc(size * sizeof(float));
	
	create_embedding(W_Q, size, -0.05, 0.05);
	create_embedding(W_K, size, -0.05, 0.05);
	create_embedding(W_V, size, -0.05, 0.05);
	
	Q = (float*)malloc(seq_len * embedding_dim * sizeof(float));
	K = (float*)malloc(seq_len * embedding_dim * sizeof(float));
	V = (float*)malloc(seq_len * embedding_dim * sizeof(float));

	//could probably change attention_scores and attention_weights buffer size to just seq_len
	//depends on what values should be analyzed	
	attention_scores = 	(float*)malloc(seq_len * seq_len  * sizeof(float)); 
	attention_weights = 	(float*)malloc(seq_len * seq_len  * sizeof(float)); 
	weighted_sums = 	(float*)malloc(seq_len * embedding_dim  * sizeof(float)); 

	for (int b = 0; b < batch_seq; ++b) {
		//project embedding - >Q,K,V
		for (int i = 0; i < seq_len; ++i) {
			for (int k = 0; k < embedding_dim; ++k) {
				int idx = i * embedding_dim + k;
				Q[idx] = 0.0f;
				K[idx] = 0.0f;
				V[idx] = 0.0f;
				for (int j = 0; j < embedding_dim; ++j) {
					int embed_idx = b * seq_len * embedding_dim + i * embedding_dim + j;
					int qkv_idx = j * embedding_dim + k;
					Q[idx] += embedding[embed_idx] * W_Q[qkv_idx];
					K[idx] += embedding[embed_idx] * W_K[qkv_idx];
					V[idx] += embedding[embed_idx] * W_V[qkv_idx];
				}
			}
		}

		//compute scores: Q * K^T
		for (int i = 0; i < seq_len; ++i) {
			for (int j = 0; j < seq_len; ++j) {
				float dot_prod = 0.0f;
				for (int k = 0; k < embedding_dim; ++k) 
					dot_prod += Q[i * embedding_dim + k] * K[j * embedding_dim + k];

				attention_scores[i * seq_length + j] = dot_prod;
			}
			/* 
			 *  softmax for attention weights
			 *  softmax normalization
			 *  max value is needed for numerical stability to prevent overflow in the exponential function
			 *
			 *  WORK IN PROGRES
			 *
			 *  GOES BY ROW
			 *
			 */

			float max_score = attention_scores[i * seq_len];
			for (int j = 1; j < seq_length; ++j) { 
			    	int idx = i * seq_len + j;
				if (attention_scores[idx] > max_score) 
				max_score = attention_scores[idx];
			}

			float score_sum_exp = 0.0f;
			for (int j = 0; j < seq_len; ++j) {
				int idx = i * seq_len + j;
				float tmp = exp(attention_scores[idx] - max_score);

				attention_weights[idx] = tmp;	
				score_sum_exp += tmp;
			}
			
			for (int j = 0; j < seq_len; ++j) {
				int idx = i * seq_len + j;
				attention_weights[idx] /= score_sum_exp;	
			}
			//eof softmax
			
			//TODO WEIGHTED_SUMS
		}
	}
}	


