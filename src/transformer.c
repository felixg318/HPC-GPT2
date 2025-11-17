#include <math.h>
#include <stdio.h>

//compute max score for softmax 
void simple_soft_attention(float *embedding, int batch_seq, int seq_length, int embedding_dim) {
	float *attention_scores = malloc(batch_seq * seq_length * seq_length  * sizeof(float));
	float *attention_weights = malloc(batch_seq * seq_length * seq_length  * sizeof(float)); 
	
	for (int b = 0; b < batch_seq; ++b) {  
		for (int i = 0; i < seq_length; ++i) {
			for (int j = 0; j < seq_length; ++j) { 
				float dot_prod = 0.0f;
				for (int d = 0; d < embedding_dim; ++d) {
					//query * input
					dot_prod += embedding[(b * seq_length + i) * embedding_dim + d] * embedding[(b * seq_length + j) * embedding_dim + d];
				}
				attention_scores[(b * seq_length + i) * seq_length + j] = dot_prod;
			}

			//softmax normalization
			float score_sum_exp = 0.0f;

			for (int j = 0; j < seq_length; ++j) {
				int idx = b * seq_length * seq_length * i + j;
				float tmp = exp(attention_scores[id]);

				attention_weights[idx] = tmp;	
				score_sum_exp += tmp;
			}
			
			for (int j = 0; j < seq_length; ++j) {
				int idx = b * seq_length * seq_length * i + j;
				attention_weights[idx] /= score_sum_exp;	
			}
			//eof softmax
		}
       	}
	free(attention_scores);
	free(attention_weights);
}
