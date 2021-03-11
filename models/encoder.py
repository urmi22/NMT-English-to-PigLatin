import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from models.grucell import MyGRUCell




class GRUEncoder(nn.Module):
    def __init__(self, device, vocab_size, opts):
        super(GRUEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = opts.hidden_size
        self.opts = opts
        self.device = device

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.gru = MyGRUCell(self.hidden_size, self.hidden_size)

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the 
        initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. 
            (batch_size x hidden_size)
        """
        return torch.zeros(bs, self.hidden_size).to(self.device)