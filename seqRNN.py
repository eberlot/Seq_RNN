#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  17 09:07:53 2020
@author: Eva + Jonathan
"""
import torch
import torch.nn as nn
import numpy as np
import random

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
  "numEpisodes": 40,
  "memPeriod": 10,
  "forceWidth": 25,
  "forceIPI": 10,
  "RT": 12,
  "cueOn": 8,
  "cueOff": 2,
  "preTime": 10} # before visual cues
simparams.update({"instTime": (simparams["cueOn"]+simparams["cueOff"])*simparams["numTargets"]+
  +simparams["cueOn"]+simparams["memPeriod"]})
simparams.update({"moveTime": (simparams["forceIPI"]*simparams["numTargets"])+simparams["forceWidth"]})
simparams.update({"trialTime": simparams["instTime"]+simparams["RT"]+simparams["moveTime"]})

# define inputs and target outputs 
def rnn_inputs_targetoutputs(simparams):
    trial_n = simparams["numEpisodes"]
    seq_data = np.zeros([trial_n,simparams["numTargets"]])
    in_data = np.zeros([simparams["trialTime"],simparams["numEpisodes"],simparams["numTargets"]+1])
    out_data = np.zeros([simparams["trialTime"],trial_n,simparams["numTargets"]])
    GoTrial = random.choices([0,1],weights=[0.2,0.8],k=simparams["numEpisodes"])
    y = gaussian()
    for i in range(trial_n):
        seq_data = np.random.randint(simparams["minTarget"],high=simparams["maxTarget"],size=[1,simparams["numTargets"]])
        t=simparams["preTime"];
        for j in range(simparams["numTargets"]): # define targets
            t_inp = range(t,t+simparams["cueOn"])
            in_data[t_inp,i,int(seq_data[0,j])] = 1
            t = t+simparams["cueOn"]+simparams["cueOff"]
        # whether go or no-go trial
        if GoTrial[i]==1: # go trial
            in_data[range(t+simparams["memPeriod"],t+simparams["memPeriod"]+
                      simparams["cueOn"]),i,simparams["numTargets"]] = 1 # go signal
            # expected output
            t=simparams["instTime"]+simparams["RT"]
            for j in range(simparams["numTargets"]):
                t_out = range(t,t+simparams["forceWidth"])
                previous = out_data[t_out,i,int(seq_data[0,j])]
                target = y
                out_data[t_out,i,int(seq_data[0,j])] = np.maximum(previous,target)
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
num_classes = 5
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

        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                          batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.zeros_(self.fc.bias)
        #nn.init.xavier_uniform_(self.fc.weight) # initialize weights
        self.h0 = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size]).requires_grad_().to(device))
        #torch.nn.init.normal_(self.rnn.weight_hh_l0, mean=0, std=1/np.sqrt(hidden_size)*1.1)
        #torch.nn.init.normal_(self.rnn.weight_ih_l0, mean=0, std=1/np.sqrt(input_size))

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        a = self.h0.repeat([1, batch_size, 1])
        out, _ = self.rnn(x, self.h0.repeat([1, batch_size, 1]))
        hidden_states = out
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out, hidden_states


# Instantiate RNN model
rnn = RNN(num_classes, input_size, output_size, hidden_size, num_layers).to(device)
print(rnn)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()
#criterion = torch.nn.BCEWithLogitsLoss()
L2_penalty = 1e-6 # 1e-6 for GRU, 1e-4 for RNN
learning_rate = 1e-3 # 1e-3 for GRU, 1e-4 for RNN
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=L2_penalty)

# Train the model
max_epochs = 50000      # maximum allowed number of iterations
loss_stop = 0.0001        # stopping criterion
loss_iter = 1           # initialized loss
epoch = 0
loss_list = []
while loss_iter>loss_stop and epoch<max_epochs:
    optimizer.zero_grad()
    [inputs, labels] = rnn_inputs_targetoutputs(simparams)
    inputs = inputs.to(device)
    labels = labels.to(device)
    # forward pass
    outputs, hidden = rnn(inputs)
    # compute the loss
    loss = criterion(outputs, labels)
    loss_iter = loss.cpu().detach().numpy()
    loss_list.append(loss_iter)
    # backpropagation
    loss.backward()
    optimizer.step()
    #for param in rnn.parameters():
     #   print(param.grad.data.sum())

    epoch +=1
    if epoch % 100 == 0 or loss_iter <= loss_stop:
        print('Epoch: {}........'.format(epoch), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        inputs = inputs.cpu()
        labels = labels.cpu()
        outputs = outputs.cpu()
        hidden = hidden.cpu()

        # visualize one training step
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3)
        plt.figure(figsize=(10, 2))
        axs[0,0].plot(inputs[:,0,:]), axs[0,0].set_title('input')
        axs[0,1].plot(labels[:,0,:]), axs[0,1].set_title('target output')
        axs[0,2].plot(outputs.detach().numpy()[:,0,:]), axs[0,2].set_title('generated output')
        axs[1,0].plot(hidden.detach().numpy()[:,0,:]), axs[1,0].set_title('hidden states')
        axs[1,1].plot(loss_list), axs[1,1].set_title('MSE')
        plt.show()
