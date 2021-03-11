import pdb
import numpy as np

from sklearn.metrics import classification_report



import torch

from utils import word_to_tensor
from operator import itemgetter






def prediction(args, val_dict, idx_dict, encoder, decoder, device):
	start_token = idx_dict['start_token']
	end_token = idx_dict['end_token']
	char_to_index = idx_dict['char_to_index']
	index_to_char = idx_dict['index_to_char']

	for source_target_length, source_target_pairs in val_dict.items():
		num_batch = int(np.ceil(len(source_target_pairs) / args.batch_size))

		for batch_idx in range(num_batch):
			start_idx = batch_idx * args.batch_size
			end_idx = start_idx + args.batch_size

			source_words = [source_target_pair[0] for source_target_pair in source_target_pairs[start_idx : end_idx]]
			target_words = [source_target_pair[1] for source_target_pair in source_target_pairs[start_idx : end_idx]]
			
			source_tensor = word_to_tensor(source_words, char_to_index, is_source = True).to(device)
			target_tensor = word_to_tensor(target_words, char_to_index, is_source = False).to(device) 

			annotations, last_hidden_state = encoder(source_tensor)
			decoder_output, _ = decoder(target_tensor, annotations, last_hidden_state)

			predict = decoder_output.argmax(dim = 2)
			target = target_tensor.reshape(target_tensor.shape[0] * target_tensor.shape[1])
			
			for i, each_tensor in enumerate(predict[:, 1:]):
				each_tensor = list(each_tensor.data.numpy())
				pred_chars = itemgetter(*each_tensor)(index_to_char)
				pred_words = ''.join(pred_chars)

				print("source word : %s\t predict word : %s"%(source_words[i], pred_words))

			# predict = predict.data.cpu().numpy()
			# target = target.data.cpu().numpy()
			# print(classification_report(target, predict))




