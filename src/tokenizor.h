#include <string>
#include <set>
#include <vector>
#include <unordered_map>

//Possibly have a vector of Tokenized objects?
class Tokenizor {
private:
	std::string fpath {};
	std::set<char> chs {};
	
	std::unordered_map<char, int> tokens;
	std::unordered_map<int, char> ids;
	
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
