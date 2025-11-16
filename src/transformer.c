#include <math.h>
//not correct
void simple_soft_attention(float *embedding, int batch_seq, int seq_length, int embedding_dim) {
  float *attention_score = malloc(batch_seq * seq_length * seq_length  * sizeof(float));
    for (int b = 0; b < batch_seq; ++b) {  
        for (int i = 0; i < seq_length; ++i) {
            for (int j = 0; j < seq_length; ++j) { 
                float dot_prod = 0.0f;
              
                for (int d = 0; d < embedding_dim; ++d) {
                    dot_prod += embedding[(b * seq_length + i) * embedding_dim + d] * embedding[(b * seq_length + j) * embedding_dim + d];
                }
                attention_score[(b * seq_length + i) * seq_length + j] = dot_prod;
            }
        }
    }
}
