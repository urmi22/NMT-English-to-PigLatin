import pdb
import argparse
import os
import numpy as np

import torch


from data_loader import LoadData
from utils import create_dict
from models.encoder import GRUEncoder
from models.decoder import GRUDecoder
from trainer import train_one_epoch
from test import prediction





def main():
	parser = argparse.ArgumentParser(description = "NMT-eng-to-pig-latin")
	parser.add_argument('--cuda', action = 'store_true', default = False, help = "use cuda")
	parser.add_argument('--nepochs', type = int, default = 10, help = "number of epochs")
	parser.add_argument('--checkpoint_dir', type = str, default = "checkpoints", help = "checkpoint directory")
	parser.add_argument('--learning_rate', type = float, default = 0.001, help = "learning rate")
	parser.add_argument('--lr_decay', type = float, default = 0.99, help = "learning decay")
	parser.add_argument('--batch_size', type = int, default = 64, help = "batch size")
	parser.add_argument('--hidden_size', type = int, default = 20, help = "hidden size")
	parser.add_argument('--seed', type = int, default = 3456, help = "seed value")
	parser.add_argument('--data_path', type = str, default = "data/pig_latin_data.txt", help = "data path")
	parser.add_argument('--split', type = float, default = 0.3, help = "validation split")

	#options: rnn / rnn_attention / transformer
	parser.add_argument('--decoder_type', type = str, default = "GRU", help = "decoder type")

	# options: additive / scaled_dot
	parser.add_argument('--attention_type', type = str, default = "additive", help = "attention type")

	args = parser.parse_args()

	use_cuda = args.cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print("using %s"%(device))

	torch.manual_seed(args.seed)

	load_data = LoadData(data_path = args.data_path)

	line_pairs, vocab_size, idx_dict = load_data.read_data()

	load_data.print_data_stats()

	total_line_pairs = len(line_pairs)
	validaton_length = int(total_line_pairs * args.split)
	train_data, validation_data = line_pairs[:-validaton_length], line_pairs[-validaton_length:]

	train_dict, validation_dict = create_dict(train_data), create_dict(validation_data)

	encoder = GRUEncoder(device, vocab_size, args)

	if args.decoder_type == "GRU":
		decoder = GRUDecoder(vocab_size, args.hidden_size)

	celoss = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = args.learning_rate)


	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	for epoch in range(args.nepochs):
		mean_loss = np.around(train_one_epoch(args, train_dict, idx_dict, encoder, decoder, celoss, optimizer, device) , 3)
		# train_one_epoch(args, validation_dict, idx_dict, encoder, decoder, celoss, optimizer, device)
		print("%d/%d --> loss : %f\n"%(epoch , args.nepochs, mean_loss))
	torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, "encoder.pt"))
	torch.save(decoder.state_dict(), os.path.join(args.checkpoint_dir, "decoder.pt"))
	
	prediction(args, validation_dict, idx_dict, encoder, decoder, device)

if __name__ == '__main__':
	main()