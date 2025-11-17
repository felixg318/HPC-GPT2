#include "tokenizor.hpp"
#include <fstream>
#include <iostream>
#include <utility>
#include <cctype>

Tokenizor::Tokenizor(){}
Tokenizor::~Tokenizor(){}

void Tokenizor::extract_tokens() {
	bool debug {true};
	
	std::ifstream infile(this->fpath, std::ios::in);
	std::string buf {""}; 

        if (!infile.is_open()) {
                std::cout << "Failed to open file" << '\n';
                return;
        }
	std::string file_contents { std::istreambuf_iterator<char>(infile), std::istreambuf_iterator<char>() };
        infile.close();

	std::ofstream myfile;
	myfile.open("debug.txt");

	std::cout<< file_contents.size() << " characters in document" << '\n';	
	
	for (size_t i = 0; i < file_contents.size(); ++i) {
		char ch = file_contents[i];
		if (isalpha(ch))
			buf += ch;
		else if (isdigit(ch))
			buf += ch;
		else if (ispunct(ch) || iscntrl(ch) || isblank(ch)) {
			this->token_list.push_back(buf);
			if (debug)
				myfile << buf;

			std::string tmp (1, ch);
//			token_list.push_back(std::string(1, ch));
			this->token_list.push_back(tmp);
			if (debug)
				myfile << tmp;
			buf.clear();
		}
	}

	std::cout << "Tokenized " << token_list.size() << " words" << '\n';
	
	myfile.close();
}

int Tokenizor::add_token(std::string token) {
	auto search = this->tokens.find(token);
	if (search == this->tokens.end()) { 
		int id = ++this->vocab_size;
		this->tokens[token] = id;
		this->ids[id] = token;
	}
	return this->vocab_size;
}

void Tokenizor::pad_data(int new_size) {
	int diff = new_size - this->data.size();
	int id = this->add_token("[PAD]");
	std::cout << "Before padding:  " << this->data.size() << "\n";
	
	for (int i = 0; i < diff; ++i) 
		this->data.push_back(id);	

	std::cout << "After padding:  " << this->data.size() << "\n";
	std::cout << "New vocab size: " << this->vocab_size << "\n";

	this->pad_id = id;
}

void Tokenizor::encode() {
	for (size_t i = 0; i < this->token_list.size(); ++i) {
		auto search = this->tokens.find(this->token_list[i]);
		if (search != this->tokens.end()) { 
			this->data.push_back(search->second);
		}
		else {
			this->tokens[token_list[i]] = this->vocab_size;
			ids[this->vocab_size] = token_list[i];
			
			this->data.push_back(this->vocab_size++);
		}	
	}
	std::cout << "Vocabulary size " << vocab_size << '\n';
	//free memory early
	this->token_list.clear();
	this->token_list.shrink_to_fit();
 		 
}

void Tokenizor::decode(int *a, int size) { 
        std::ofstream myfile;
        myfile.open("out.txt");
    
        for (int i = 0; i < size; ++i) {
		auto search = this->ids.find(a[i]);
                if (search != this->ids.end()) {
			std::string str = search->second; //can be remove if not used in the future
                        myfile << str;
                }
        }

        myfile.close();
}

void Tokenizor::set_file(std::string fpath) { this->fpath = fpath;}
int Tokenizor::get_pad_id() { return this->pad_id;}
void Tokenizor::print_data() {
	for (size_t i = 0; i < this->data.size(); ++i)
		std::cout << this->data[i] << ' ';
}
