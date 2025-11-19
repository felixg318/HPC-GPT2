#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 *	THIS FILE IS A WORK IN PROGRESS!!!
 *
 */

void apply_casual_attention_mask_v1(float *scores, int rows, int cols);
void apply_softmask_v1(float *scores, float *weights, int rows, int cols);
void save_to_file(float *mat, int dim1, int dim2, int dim3, const char *filename);
void save_2d_to_file(float *mat, int rows, int cols, const char *filename);

void self_attention_v1(float *embedding,
			float *context_vec,
	       		float *W_Q,
			float *W_V,
			float *W_K,	
			int num_seq, 
			int seq_len, 
			int embedding_dim, 
			int in_dim, 
			int out_dim) {
	
	if (embedding_dim != in_dim) {
		printf("Bad dimensions for self_attention_forward_v1. Embedding dimension does not match in_dim for weight matrices\n");
		return;
	}
	
	float *Q, *K, *V;
	float *attention_scores, *attention_weights;

	Q = (float*)malloc(seq_len * embedding_dim * sizeof(float));
	K = (float*)malloc(seq_len * embedding_dim * sizeof(float));
	V = (float*)malloc(seq_len * embedding_dim * sizeof(float));

	//could probably change attention_scores and attention_weights buffer size to just seq_len
	//depends on what values should be analyzed	
	attention_scores = 	(float*)malloc(seq_len * seq_len  * sizeof(float)); 
	attention_weights = 	(float*)malloc(seq_len * seq_len  * sizeof(float)); 

	for (int s = 0; s < num_seq; ++s) {
		//project embedding - >Q,K,V
		for (int i = 0; i < seq_len; ++i) {
			for (int k = 0; k < out_dim; ++k) {
				int idx = i * embedding_dim + k;
				Q[idx] = 0.0f;
				K[idx] = 0.0f;
				V[idx] = 0.0f;
				for (int j = 0; j < embedding_dim; ++j) {
					//scaled dot prod
					int embed_idx = s * seq_len * embedding_dim + i * embedding_dim + j;
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

				attention_scores[i * seq_len + j] = dot_prod;
			
			}
		}

		apply_casual_attention_mask_v1(attention_scores, seq_len, seq_len);
		apply_softmask_v1(attention_scores, attention_weights, seq_len, seq_len); 
		
		//compute and assign to context_vec
		//Attn Weights * V
		for (int i = 0; i < seq_len;  ++i) {
			for (int d = 0; d < embedding_dim; ++d) {
				float weighted_sum = 0.0f;
				for(int j = 0; j < seq_len; ++j) {
					int w_idx = i * seq_len + j;
					int v_idx = j * embedding_dim + d;

					weighted_sum += attention_weights[w_idx] * V[v_idx];
				}
				context_vec[s * seq_len * embedding_dim + i * embedding_dim + d] = weighted_sum;

			}
		}
	}
	
	free(Q);
	free(K);
	free(V);
	free(attention_scores);
	free(attention_weights);
}	

void apply_casual_attention_mask_v1(float *scores, int rows, int cols) {
	for(int i = 0; i < rows; ++i) 
		for(int j = i + 1; j < cols; ++j) 
			scores[i * cols + j] = -INFINITY; 
}
/* 
 *  softmax for attention weights
 *  softmax normalization
 *  max value is needed for numerical stability to prevent overflow in the exponential function
 *
 */
void apply_softmask_v1(float *scores, float *weights, int rows, int cols) {
	for (int i = 0; i < rows; ++i) { 	
		//process by row
		float max_score = scores[i * cols];
		for (int j = 1; j < cols; ++j) { 
			int idx = i * cols + j;
			if (scores[idx] > max_score) 
				max_score = scores[idx];
		}
		float score_sum_exp = 0.0f;
		for (int j = 0; j < cols; ++j) {
			int idx = i * cols + j;
			float tmp = exp(scores[idx] - max_score);

			weights[idx] = tmp;	
			score_sum_exp += tmp;
		}
		
		for (int j = 0; j < cols; ++j) {
			int idx = i * cols + j;
			weights[idx] /= score_sum_exp;	
		}
	}
}
/*
 *
 *
 *  simple attention mechanism. was done for learning
 *
 *
 *
 *
void simple_soft_attention(float *embedding, int batch_seq, int seq_length, int embedding_dim) {
	float *attention_scores = (float*)malloc(seq_length * seq_length  * sizeof(float));
	float *attention_weights = (float*)malloc(seq_length * seq_length  * sizeof(float)); 
	float *weighted_sums = (float*)malloc(seq_length * embedding_dim  * sizeof(float)); 
	
	for (int i = 0; i < seq_length; ++i) {
		for (int j = 0; j < seq_length; ++j) { 
			float dot_prod = 0.0f;
			for (int d = 0; d < embedding_dim; ++d) {
				int query_idx = i * embedding_dim + d;
				int key_idx = j * embedding_dim + d;

				dot_prod += embedding[query_idx] * embedding[key_idx];
			}
			attention_scores[ i * seq_length + j] = dot_prod;
		}
		//softmax normalization
		//max value is needed for numerical stability to prevent overflow in the exponential function
		float max_score = attention_scores[i * seq_length];
		for (int j = 1; j < seq_length; ++j) {
		    int idx = i * seq_length + j;
		    if (attention_scores[idx] > max_score) {
			max_score = attention_scores[idx];
		    }
		}

		float score_sum_exp = 0.0f;

		for (int j = 0; j < seq_length; ++j) {
			int idx = i * seq_length + j;
			float tmp = exp(attention_scores[idx] - max_score);

			attention_weights[idx] = tmp;	
			score_sum_exp += tmp;
		}
		
		for (int j = 0; j < seq_length; ++j) {
			int idx = i * seq_length + j;
			attention_weights[idx] /= score_sum_exp;	
		}
		//eof softmax
		
		//attention weights * input embeddings
		for (int d = 0; d < embedding_dim; ++d) {
			float weighted_sum = 0.0f;
			for(int j = 0; j < seq_length; ++j) {
				int weight_idx = i * seq_length    + j;  
				int embed_idx =  j * embedding_dim + d;
				weighted_sum += attention_weights[weight_idx] * embedding[embed_idx];
			}
			weighted_sums[i * embedding_dim + d] = weighted_sum;
		}
	}
	
	free(attention_scores);
	free(attention_weights);
	free(weighted_sums);
}
*/

