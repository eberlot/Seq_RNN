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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.manual_seed(777)

# general specifications
simparams =	{
  "numTargets": 5,
  "minTarget": 1,
  "maxTarget": 5,
  "numEpisodes": 10,
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
        inputs = torch.from_numpy(in_data)
        # define output
        out_data = np.zeros([simparams["trialTime"],trial_n,simparams["numTargets"]])
        t=simparams["instTime"]+simparams["RT"];
        for j in range(simparams["numTargets"]):
            t_out = range(t,t+simparams["forceWidth"])
            previous = out_data[t_out,i,int(seq_data[i,j])]
            target = y;
            out_data[t_out,i,int(seq_data[i,j])] = np.maximum(previous,target);
            t = t+simparams["forceIPI"]
        
        outputs = torch.from_numpy(out_data)
    return inputs.float(), outputs.float()
       
def gaussian():
    x = np.arange(-12.5, 12.5, 1)
    s = 3
    y = 1./np.sqrt(2.*np.pi*s**2) * np.exp(-x**2/(2.*s**2))
    y = y/np.max(y)
    return y

[inputs,labels] = rnn_inputs_outputs(simparams)
inputs.to(device)
labels.to(device)

num_classes = 5
input_size = inputs.shape[2]
output_size = labels.shape[2]
hidden_size = 100  # number of units
batch_size = simparams["numEpisodes"]
sequence_length = inputs.shape[0] 
num_layers = 1  # one-layer rnn

class RNN(nn.Module):

    def __init__(self, num_classes, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        #self.rnn = nn.RNN(input_size=5, hidden_size=5, batch_first=True)
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=False,
                          nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out)
        #out = nn.Sigmoid(out)
        return out


# Instantiate RNN model
rnn = RNN(num_classes, input_size, output_size, hidden_size, num_layers)
print(rnn)
rnn.to(device)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Train the model
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    [inputs, labels] = rnn_inputs_outputs(simparams)
    inputs.to(device)
    labels.to(device)
    outputs = rnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {}/{}........'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
    
