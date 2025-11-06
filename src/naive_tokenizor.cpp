#include "tokenizor.h"
#include <fstream>
#include <iostream>
#include <utility>

Tokenizor::Tokenizor(){}
Tokenizor::~Tokenizor(){}

void Tokenizor::extract_chars() {
	std::string strin {};
	std::ifstream infile(this->fpath, std::ios::in);

	if (!infile.is_open()) {
		std::cout << "Failed to open file" << '\n';
		return;
	}
	while (std::getline(infile, strin)) {
		for (int i = 0; i < strin.size(); i++){
			auto insert_retval = this->chs.insert(strin[i]);
			if (std::get<1>(insert_retval))
				this->vocab_size++;
		}
	}

	infile.close();
}

void Tokenizor::encode() {}
void Tokenizor::decode() {}

void Tokenizor::set_file(std::string fpath) { this->fpath = fpath;}
void Tokenizor::print_set() {
	std::cout << "Set elements:" << '\n';
	std::cout << "{\n";
	for (const auto& elm : this->chs)
		std::cout << elm << ' ';
	std::cout << "\n}\n";
	std::cout << "Vocab size: " << this->vocab_size <<'\n';
}
