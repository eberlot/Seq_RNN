#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  17 09:07:53 2020
@author: Eva
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

torch.manual_seed(777)

# general specifications
simparams =	{
  "numTargets": 5,
  "minTarget": 1,
  "maxTarget": 5,
  "numEpisodes": 1,
  "memPeriod": 10,
  "forceWidth": 25,
  "forceIPI": 10,
  "RT": 12,
  "cueOn": 8,
  "cueOff": 2,
  "preTime": 10} # before visual cues
simparams.update({"instTime": (simparams["cueOn"]+simparams["cueOff"])*simparams["numTargets"]+\
  +simparams["cueOn"]+simparams["memPeriod"]})
simparams.update({"moveTime": (simparams["forceIPI"]*simparams["numTargets"])+simparams["forceWidth"]})
simparams.update({"trialTime": simparams["instTime"]+simparams["RT"]+simparams["moveTime"]})

# define inputs and outputs
def rnn_inputs_outputs(simparams):
    trial_n = simparams["numEpisodes"]
    seq_data = np.zeros([trial_n,simparams["numTargets"]])
    y = gaussian()
    for i in range(trial_n):
        seq_data[i,:] = np.random.randint(simparams["minTarget"],high=simparams["maxTarget"],size=[1,simparams["numTargets"]])
        # define input stimulus presentation
        in_data = np.zeros([simparams["trialTime"],simparams["numEpisodes"],simparams["numTargets"]+1])
        t=simparams["preTime"];
        for j in range(simparams["numTargets"]):
            t_inp = range(t,t+simparams["cueOn"])
            in_data[t_inp,i,int(seq_data[i,j])] = 1;
            t = t+simparams["cueOn"]+simparams["cueOff"]
        in_data[range(t+simparams["memPeriod"],t+simparams["memPeriod"]+\
                      simparams["cueOn"]),i,simparams["numTargets"]] = 1; # for go signal    
        inputs = Variable(torch.Tensor(in_data))
        # define output
        out_data = np.zeros([simparams["numTargets"],simparams["trialTime"]])
        t=simparams["instTime"]+simparams["RT"];
        for j in range(simparams["numTargets"]):
            t_out = range(t,t+simparams["forceWidth"])
            previous = out_data[int(seq_data[i,j]),t_out]
            target = y;
            out_data[int(seq_data[i,j]),t_out] = np.maximum(previous,target); 
            t = t+simparams["forceIPI"]
        
        outputs = Variable(torch.Tensor(out_data))     
    return inputs,outputs
       
def gaussian():
    x = np.arange(-12.5, 12.5, 1);
    s = 3;
    y = 1./np.sqrt(2.*np.pi*s**2) * np.exp(-x**2/(2.* s**2))
    y = y/np.max(y)
    return y

[inputs,labels] = rnn_inputs_outputs(simparams)


num_classes = 5
input_size = inputs.shape[2]
hidden_size = 100  # number of units
batch_size = 1  
sequence_length = inputs.shape[0] 
num_layers = 1  # one-layer rnn

class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        #self.rnn = nn.RNN(input_size=5, hidden_size=5, batch_first=True)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.h_0 = self.initialize_hidden(hidden_size)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        x = x.unsqueeze(0)
        self.rnn.flatten_parameters()
        out, self.h_0 = self.rnn(x, self.h_0)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out
    
    def initialize_hidden(self, rnn_hidden_size):
        # n_layers * n_directions, batch_size, rnn_hidden_size
        return Variable(torch.randn(1,1,rnn_hidden_size),
                        requires_grad=True)


# Instantiate RNN model
rnn = RNN(num_classes, input_size, hidden_size, num_layers)
print(rnn)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    outputs = rnn(rnn)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
