#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 *	THIS FILE IS A WORK IN PROGRESS!!!
 *	We could make a Multihead Attention but it adds complexity
 *	Lets stick with casual attention and prioritize MPI + CUDA
 *
 */

void apply_casual_mask(float *scores, int rows, int cols);
void apply_dropout(float *attention_weights, int size, int dropout_rate);
void apply_softmask_v1(float *scores, float *weights, int rows, int cols);
void save_to_file(float *mat, int dim1, int dim2, int dim3, const char *filename);
void save_2d_to_file(float *mat, int rows, int cols, const char *filename);

//pack these variables into a struct
void self_attention_v1(float *embedding,
			float *context_vec,
	       		float *W_Q,
			float *W_V,
			float *W_K,	
			int num_seq, 
			int seq_len, 
			int embedding_dim, 
			int in_dim, 
			int out_dim,
			int is_training) {
	
	if (embedding_dim != in_dim) {
		printf("Bad dimensions for self_attention_forward_v1. Embedding dimension does not match in_dim for weight matrices\n");
		return;
	}

	int dropout_rate = 0.2f; //pack into a struct
	
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

		apply_casual_mask(attention_scores, seq_len, seq_len);
		apply_softmask_v1(attention_scores, attention_weights, seq_len, seq_len); 
		
		if (is_training)
			apply_dropout(attention_weights, seq_len * seq_len, dropout_rate);

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

void apply_casual_mask(float *scores, int rows, int cols) {
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

void apply_dropout(float *weights, int size, int dropout_rate) {
    float keep_prob = 1.0f - dropout_rate;
    float scale = 1.0f / keep_prob;
    
    for (int i = 0; i < size; ++i) {
        float rand_val = (float)rand() / RAND_MAX;
        
        if (rand_val < dropout_rate) 
            weights[i] = 0.0f;
        else 
            weights[i] *= scale;
    }		
}
