#include "tokenizor.h"
#include <fstream>
#include <iostream>
#include <utility>

Tokenizor::Tokenizor(){}
Tokenizor::~Tokenizor(){}

void Tokenizor::extract_chars() {
        std::ifstream infile(this->fpath, std::ios::in);
        char ch; 
        int id {0};

        if (!infile.is_open()) {
                std::cout << "Failed to open file" << '\n';
                return;
        }
        while (infile.get(ch)) {
                auto insert_retval = this->chs.insert(ch);
                if (std::get<1>(insert_retval))
                        this->vocab_size++;
        }   
        infile.close();

        for (auto it = this->chs.begin(); it != this->chs.end(); ++it) {
                tokens[*it] = id; 
                ids[id++] = *it;
        }   

}

void Tokenizor::encode() {
        std::ifstream infile(this->fpath, std::ios::in);
        char ch;

        if (!infile.is_open()) {
                std::cout << "Failed to open file" << '\n';
                return;
        }   
        while (infile.get(ch)) {
                if (auto search = this->tokens.find(ch); search != this->tokens.end())
                        this->data.push_back(search->second);   
                else
                        std::cout << "Token not found\n";           
        }   

        infile.close();
}

void Tokenizor::decode() {
        std::ofstream myfile;
        myfile.open("out.txt");
    
        for (int i = 0; i < this->data.size(); ++i) {
                if (auto search = this->ids.find(this->data[i]); search != this->ids.end()) {
                        char c = search->second; //can be remove if not used in the future
                        myfile << c;
                }
        }

        myfile.close();
}

void Tokenizor::set_file(std::string fpath) { this->fpath = fpath;}
void Tokenizor::print_set() {
        std::cout << "Set elements:" << '\n';
        std::cout << "{\n";
        for (const auto& elm : this->chs)
                std::cout << elm << ' ';
        std::cout << "\n}\n";
        std::cout << "Vocab size: " << this->vocab_size <<'\n';
}
