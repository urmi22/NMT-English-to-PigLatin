import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # ------------
        # FILL THIS IN
        # ------------
        ## Input linear layers
        self.Wiz = nn.Linear(in_features = self.input_size, \
                             out_features = self.hidden_size)
        self.Wir = nn.Linear(in_features = self.input_size, \
                             out_features = self.hidden_size)
        self.Wih = nn.Linear(in_features = self.input_size, \
                             out_features = self.hidden_size)

        ## Hidden linear layers
        self.Whz = nn.Linear(in_features = self.hidden_size, \
                             out_features = self.hidden_size)
        self.Whr = nn.Linear(in_features = self.hidden_size, \
                             out_features = self.hidden_size)
        self.Whh = nn.Linear(in_features = self.hidden_size, \
                             out_features = self.hidden_size)
        


    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN
        # ------------
        z = F.sigmoid(self.Wiz(x) + self.Whz(h_prev))
        r = F.sigmoid(self.Wir(x) + self.Whr(h_prev))
        g = F.tanh(self.Wih(x) + r * self.Whh(h_prev))
        h_new = (1 -z) * g + z * h_prev
        return h_new