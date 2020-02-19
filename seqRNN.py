#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  17 09:07:53 2020
@author: Eva
"""
import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.manual_seed(777)

# general specifications of training simulations, trial structure
simparams =	{
  "numTargets": 5,
  "minTarget": 1,
  "maxTarget": 5,
  "numEpisodes": 50,
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

# define inputs and target outputs 
def rnn_inputs_targetoutputs(simparams):
    trial_n = simparams["numEpisodes"]
    seq_data = np.zeros([trial_n,simparams["numTargets"]])
    in_data = np.zeros([simparams["trialTime"],simparams["numEpisodes"],simparams["numTargets"]+1])
    out_data = np.zeros([simparams["trialTime"],trial_n,simparams["numTargets"]])
    y = gaussian()
    for i in range(trial_n):
        seq_data = np.random.randint(simparams["minTarget"],high=simparams["maxTarget"],size=[1,simparams["numTargets"]])
        t=simparams["preTime"];
        for j in range(simparams["numTargets"]): # define targets
            t_inp = range(t,t+simparams["cueOn"])
            in_data[t_inp,i,int(seq_data[0,j])] = 1;
            t = t+simparams["cueOn"]+simparams["cueOff"]
        # go signal    
        in_data[range(t+simparams["memPeriod"],t+simparams["memPeriod"]+\
                      simparams["cueOn"]),i,simparams["numTargets"]] = 1;     
        # define expected output
        t=simparams["instTime"]+simparams["RT"];
        for j in range(simparams["numTargets"]):
            t_out = range(t,t+simparams["forceWidth"])
            previous = out_data[t_out,i,int(seq_data[0,j])]
            target = y;
            out_data[t_out,i,int(seq_data[0,j])] = np.maximum(previous,target);
            t = t+simparams["forceIPI"]  
    inputs = torch.from_numpy(in_data)
    target_outputs = torch.from_numpy(out_data)
    return inputs.float(), target_outputs.float()
       
# convolve expected output force profile with a Gaussian window - for now hard-coded
def gaussian():
    x = np.arange(-12.5, 12.5, 1)
    s = 3
    y = 1./np.sqrt(2.*np.pi*s**2) * np.exp(-x**2/(2.*s**2))
    y = y/np.max(y)
    return y

[inputs,target_outputs] = rnn_inputs_targetoutputs(simparams)
inputs = inputs.to(device)
target_outputs = target_outputs.to(device)

# here RNN specifications
num_classes = 10
input_size = inputs.shape[2]
output_size = target_outputs.shape[2]
hidden_size = 300  # number of units
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
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                          nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


# Instantiate RNN model
rnn = RNN(num_classes, input_size, output_size, hidden_size, num_layers)
print(rnn)
rnn = rnn.to(device)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()
L2_penalty = 1e-5 # L2-type regularization on weights
learning_rate = 1e-3
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=L2_penalty)

# Train the model
max_epochs = 1000000    # incorporate as a fail-safe
loss_stop = 1e-10       # stopping criterion
loss_step = 1           # initialized loss
epoch = 0
while loss_step>loss_stop:
    optimizer.zero_grad()
    [inputs, labels] = rnn_inputs_targetoutputs(simparams)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = rnn(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    loss = loss.cpu()
    loss_step = loss.detach().numpy()
    optimizer.step()
    epoch +=1
    if epoch % 50 == 0:
        print('Epoch: {}........'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        
#outputs.detach().numpy()    