#include <string>
#include <vector>
#include <unordered_map>

class Tokenizor {
private:
	int pad_id;

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
	void pad_data(int new_size);
	int get_pad_id();
	
	void encode();
	void decode(int*, int);

	int add_token(std::string);

	void set_file(std::string);
	
	void print_data();

};
