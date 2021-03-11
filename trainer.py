import pdb
import numpy as np



import torch

from utils import word_to_tensor




def train_one_epoch(args, train_dict, idx_dict, encoder, decoder, celoss, optimizer, device):
	start_token = idx_dict['start_token']
	end_token = idx_dict['end_token']
	char_to_index = idx_dict['char_to_index']

	losses = []

	for source_target_length, source_target_pairs in train_dict.items():
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

			decoder_output = decoder_output.reshape(decoder_output.shape[0] * decoder_output.shape[1], decoder_output.shape[2])
			target_tensor = target_tensor.reshape(target_tensor.shape[0] * target_tensor.shape[1])
			loss = celoss(decoder_output, target_tensor)

			losses.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	mean_loss = np.mean(losses)
	return mean_loss




