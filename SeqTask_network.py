# SeqTask_network
# This module contains pytorch-based neural networks for training Sequential finger tapping task.
# Programmers: Eva + Jonathan + Mehrdad
# Date: Aug 1st, 2020

# Importing Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_recurrent(nn.Module):
    def __init__(self, netparams):
        super(simple_recurrent, self).__init__()

        self.num_classes = netparams["num_classes"]
        self.num_layers = netparams["num_layers"]
        self.input_size = netparams["input_size"]
        self.output_size = netparams["output_size"]
        self.hidden_size = netparams["hidden_size"]
        self.sequence_length = netparams["sequence_length"]
        self.device = netparams["device"]
        self.batch_size = netparams["batch_size"]
        if netparams["RNN_type"] == "GRU":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bias=True, batch_first=False)
        elif netparams["RNN_type"] == "RNN":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bias=True, batch_first=False)
        # Manual initialization of the network (Default values seem ok, why bother?)
        #nn.init.normal_(self.rnn.weight_hh_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.weight_ih_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.bias_hh_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.bias_ih_l0, mean=0, std=1e-4)

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # Manual initialization of the network
        #nn.init.normal_(self.fc.weight, mean=0, std=1 / (np.sqrt(self.hidden_size)))
        self.h0 = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size]).requires_grad_().to(self.device))


    def forward(self, x, batch_size):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        out, _ = self.rnn(x, self.h0.repeat([1, batch_size, 1]))
        hidden_states = out
        out = self.fc(out)
        return out, hidden_states


class simple_recurrent_non_linear(nn.Module):
    def __init__(self, netparams):
        super(simple_recurrent_non_linear, self).__init__()

        self.num_classes = netparams["num_classes"]
        self.num_layers = netparams["num_layers"]
        self.input_size = netparams["input_size"]
        self.output_size = netparams["output_size"]
        self.hidden_size = netparams["hidden_size"]
        self.sequence_length = netparams["sequence_length"]
        self.device = netparams["device"]
        self.batch_size = netparams["batch_size"]
        if netparams["RNN_type"] == "GRU":
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bias=True, batch_first=False)
        elif netparams["RNN_type"] == "RNN":
            self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                              bias=True, batch_first=False)
        # Manual initialization of the network (Default values seem ok, why bother?)
        #nn.init.normal_(self.rnn.weight_hh_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.weight_ih_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.bias_hh_l0, mean=0, std=1e-4)
        #nn.init.normal_(self.rnn.bias_ih_l0, mean=0, std=1e-4)

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # Manual initialization of the network
        # Initial values for bias
        # nn.init.constant(self.fc.bias, 1)
        #nn.init.normal_(self.fc.weight, mean=0, std=1 / (np.sqrt(self.hidden_size)))
        self.h0 = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size]).requires_grad_().to(self.device))


    def forward(self, x, batch_size):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        out, _ = self.rnn(x, self.h0.repeat([1, batch_size, 1]))
        hidden_states = out
        out = F.elu(self.fc(out), alpha=0.01)
        return out, hidden_states