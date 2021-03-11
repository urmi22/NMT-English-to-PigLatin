import pdb
from collections import defaultdict
from operator import itemgetter


import torch




def create_dict(pairs):
	unique_pairs = list(set(pairs)) # Find all unique (source, target) pairs
	d = defaultdict(list)
	for (s, t) in unique_pairs:
		d[(len(s), len(t))].append((s,t))

	# pdb.set_trace()
	return d


def word_to_tensor(words, char_to_index, is_source = True):
	if is_source == True:
		tensor = torch.zeros(len(words), len(words[0]) + 1, dtype = int)
		# char_indexes = [char_to_index[word] for word in words]
		for i, word in enumerate(words):
			char_indexes = [char_to_index[char] for char in list(word)]
			tensor[i][:-1] = torch.LongTensor(char_indexes)
			tensor[i][-1] = char_to_index['EOS']
	else:
		tensor = torch.zeros(len(words), len(words[0]) + 1, dtype = int)
		for i, word in enumerate(words):
			char_indexes = list(itemgetter(*list(word))(char_to_index))
			tensor[i][0] = char_to_index['SOS']
			tensor[i][1:] = torch.LongTensor(char_indexes)
			
		

	return tensor



