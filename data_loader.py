import pdb



class LoadData():

	def __init__(self, data_path, split = 0.3):
		self.data_path = data_path
		self.split = split

	def read_file(self):
		source_words, target_words = [], []
		with open(self.data_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip()
				if line:
					source, target = line.strip().split(" ")
					source_words.append(source)
					target_words.append(target)

		return source_words, target_words


	def all_alpha_or_dash(self, s):
		"""Helper function to check whether a string is alphabetic, allowing dashes '-'.
	    """
		return all(c.isalpha() or c == '-' for c in s)
	    
    	

	def filter_lines(self, lines):
		"""Filters lines to consist of only alphabetic characters or dashes "-".
	    """
		return [line for line in lines if self.all_alpha_or_dash(line)]
	    
    	

	
	def read_data(self):
		source_words, target_words = self.read_file()
		filtered_source_words = self.filter_lines(source_words)
		filtered_target_words = self.filter_lines(target_words)

		all_characters = set(''.join(filtered_source_words)) | set(''.join(filtered_target_words))
		char_to_index = { char: index for (index, char) in enumerate(sorted(list(all_characters))) }
		start_token = len(char_to_index)
		end_token = len(char_to_index) + 1
		char_to_index['SOS'] = start_token
		char_to_index['EOS'] = end_token

		# Create the inverse mapping, from indexes to characters (used to decode the model's predictions)
		index_to_char = { index: char for (char, index) in char_to_index.items() }

		# Store the final size of the vocabulary
		vocab_size = len(char_to_index)

		line_pairs = list(set(zip(filtered_source_words, filtered_target_words)))  # Python 3

		idx_dict = {'char_to_index': char_to_index, 
			'index_to_char': index_to_char, 
			'start_token': start_token,
			'end_token': end_token}
		return line_pairs, vocab_size, idx_dict

	def print_data_stats(self):
		line_pairs, vocab_size, idx_dict = self.read_data()
		print('=' * 80)
		print('Data Stats'.center(80))
		print('-' * 80)
		for pair in line_pairs[:5]:
			print(pair)
		print('Num unique word pairs: {}'.format(len(line_pairs)))
		print('Vocabulary: {}'.format(idx_dict['char_to_index'].keys()))
		print('Vocab size: {}'.format(vocab_size))
		print('=' * 80)



if __name__ == '__main__':
	load_data = LoadData(data_path = "data/pig_latin_data.txt")
	load_data.print_data_stats()

		