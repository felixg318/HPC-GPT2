#include <string>
#include <unordered_set>
#include <vector>

class Tokenizor {
private:
	std::string fpath {};
	std::unordered_set<char> chs {};
	int vocab_size {0};
	std::vector<int> data {};


public:
	Tokenizor();
	~Tokenizor();

	void extract_chars();
	void encode();
	void decode();

	void set_file(std::string fpath);
	
	void print_set();
	void print_data();
};
