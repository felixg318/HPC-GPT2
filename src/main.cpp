#include <iostream>
#include "tokenizor.h"

int main(int argc, char** argv) {
	Tokenizor tokenizor {};

	if (argc < 2 || argc > 2) {
		std::cout << "Bad args" << '\n';
		return -1;	
	}
	
	tokenizor.set_file(argv[1]);
	tokenizor.extract_chars();

	tokenizor.print_set();

	return 0;
}
