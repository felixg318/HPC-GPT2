#include <string>
#include <vector>
#include <unordered_map>

class Tokenizor {
public:
	std::string fpath {};
	
	std::unordered_map<std::string, int> tokens;
	std::unordered_map<int, std::string> ids;
	
	int vocab_size {0};

	std::vector<int> data {};
	std::vector<std::string> token_list;

	Tokenizor();
	~Tokenizor();

	void extract_tokens();
	
	void encode();
	void decode();

	void set_file(std::string);
	
	void print_data();

};
