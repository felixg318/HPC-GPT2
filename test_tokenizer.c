#include <stdio.h>
#include "tokenizer.h"

int main(void) {
    Tokenizer tok;
    tokenizer_init(&tok, "dummy_data.txt");
    if (!tokenizer_extract(&tok)) {
        printf("tokenizer failed to extract tokens\n");
        return 1;
    }
    if (!tokenizer_encode(&tok)) {
        printf("tokenizer failed to encode tokens\n");
        tokenizer_free(&tok);
        return 1;
    }
    tokenizer_pad_to(&tok, 64);

    printf("File path: %s\n", tok.file_path ? tok.file_path : "(null)");
    printf("Text length: %zu\n", tok.text_len);
    printf("Vocabulary size: %d\n", tok.vocab_size);
    printf("Number of encoded tokens: %d\n", tok.encoded_count);
    printf("Pad token id: %d\n", tok.pad_id);

    printf("First 20 encoded ids:\n");
    for (int i = 0; i < tok.encoded_count && i < 20; ++i) {
        printf("%d ", tok.encoded[i]);
    }
    printf("\n");

    tokenizer_free(&tok);
    return 0;
}
