# Process Simplified
* Pass raw text (.txt) in first arguement
* Raw text is passed to Tokenizor object that extracts tokens and holds a vocabulary mapping
* Convert text into tokens
* Embed tokens and add positional data for the model to learn
* Create batches of token sequences with its embedding
* Pass tokens to attention mechanism (casual attention)
