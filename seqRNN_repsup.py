#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:23:14 2020
Repetition supression code

@author: Eva
"""

import torch
import torch.nn as nn
import numpy as np
import random
import itertools
import os
#import pickle
import scipy.io
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.manual_seed(777)

baseDir = '/Users/Eva/Documents/Data/Seq_RNN'
# specifications for training parameters
trainparams = {
        "maxIter": 50000,
        "lossStop": 5e-5,
        "plotOn": 1,
        "L2": 1e-6,
        "learningRate": 1e-2}

# general specifications for training simulations, trial structure
# options for type: train_goonly, train_gonogo, test_goonly'
def generate_seqdata_simparams(type):
    simparams =	{
            "numTargets": 5,
            "minTarget": 1,
            "maxTarget": 5,
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
   
    x=list(itertools.permutations([1,2,3,4,5]))
    seqdata = random.sample(x, len(x))
    if "train" in type:
        simparams.update({"numEpisodes":40})
    elif "test" in type:
        simparams.update({"numEpisodes":1})
    if "nogo" in type:
        simparams.update({"GoTrial":0.8})
    elif "only" in type:
        simparams.update({"GoTrial":1})
    return seqdata, simparams 
            
# define inputs and target outputs 
def rnn_inputs_targetoutputs(seqdata,simparams):
    trial_n = simparams["numEpisodes"]
    in_data = np.zeros([simparams["trialTime"],simparams["numEpisodes"],simparams["numTargets"]+1])
    out_data = np.zeros([simparams["trialTime"],trial_n,simparams["numTargets"]])
    GoTrial = random.choices([0,1],weights=[1-simparams["GoTrial"],simparams["GoTrial"]],k=simparams["numEpisodes"])
    y = gaussian()
    for i in range(trial_n):
        t=simparams["preTime"];
        for j in range(simparams["numTargets"]): # define targets
            t_inp = range(t,t+simparams["cueOn"])
            in_data[t_inp,i,seqdata[i][j]-1] = 1
            t = t+simparams["cueOn"]+simparams["cueOff"]
        # whether go or no-go trial
        if GoTrial[i]==1: # go trial
            in_data[range(t+simparams["memPeriod"],t+simparams["memPeriod"]+
                      simparams["cueOn"]),i,simparams["numTargets"]] = 1 # go signal
            # expected output
            t=simparams["instTime"]+simparams["RT"]
            for j in range(simparams["numTargets"]):
                t_out = range(t,t+simparams["forceWidth"])
                previous = out_data[t_out,i,seqdata[i][j]-1]
                target = y
                out_data[t_out,i,seqdata[i][j]-1] = np.maximum(previous,target)
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

seqdata,simparams = generate_seqdata_simparams("train_gonogo")
[inputs,target_outputs] = rnn_inputs_targetoutputs(seqdata,simparams)
inputs = inputs.to(device)
target_outputs = target_outputs.to(device)

# here RNN specifications
num_classes = 5
input_size = inputs.shape[2]
output_size = target_outputs.shape[2]
hidden_size = 300  # number of units
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
        self.fc = nn.Linear(hidden_size, output_size)
        #self.fc = nn.Linear(hidden_size, output_size, bias=True)
        #nn.init.zeros_(self.fc.bias)
        self.h0 = nn.Parameter(torch.zeros([self.num_layers, self.hidden_size]).requires_grad_().to(device))

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        batch_size = x.shape[1] # based on the input
        a = self.h0.repeat([1, batch_size, 1])
        out, _ = self.rnn(x, self.h0.repeat([1, batch_size, 1]))
        hidden_states = out
        out = self.fc(out)
        #out = torch.sigmoid(out)
        return out, hidden_states

def rnn_step(simparams,seqdata):
# here just make one step - used often
    #batch_size = simparams["numEpisodes"]
    [inputs, labels] = rnn_inputs_targetoutputs(seqdata,simparams)
    inputs = inputs.to(device)
    labels = labels.to(device)
    # forward pass
    outputs, hidden = rnn(inputs)
    # compute the loss
    loss = criterion(outputs, labels)
    return loss,outputs,hidden,labels,inputs

# Instantiate RNN model
rnn = RNN(num_classes, input_size, output_size, hidden_size, num_layers).to(device)
print(rnn)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()
L2_penalty = trainparams["L2"] # 1e-6 for GRU, 1e-4 for RNN
learning_rate = trainparams["learningRate"] # 1e-2 for GRU, 1e-4 for RNN
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, weight_decay=L2_penalty)

# initiate variables for training
max_epochs = trainparams["maxIter"] # maximum allowed number of iterations
loss_stop = trainparams["lossStop"] # stopping criterion
loss_iter = 1                       # initialized loss
epoch = 0
loss_list = []
epoch_test = 0
# Train the model
while loss_iter>loss_stop and epoch<max_epochs:    
    if epoch % 50 !=0: # normal training
        optimizer.zero_grad()
        seqdata,simparams = generate_seqdata_simparams("train_gonogo")
        loss,outputs,hidden,labels,inputs = rnn_step(simparams,seqdata) # one step forward
        loss_iter = loss.cpu().detach().numpy()
        loss_list.append(loss_iter)
        # backpropagation
        loss.backward()
        optimizer.step()
        epoch +=1
    else:    # every 50 steps do repetition test
        # first step for one stimulus
        optimizer.zero_grad()
        seqdata,simparams = generate_seqdata_simparams("test_gonogo")
        loss,outputs_1,hidden_1,labels,inputs = rnn_step(simparams,seqdata) # one step forward
        loss.backward()
        optimizer.step()
        # record first trial
        epoch +=1
        # now test all possible stimuli as second stimulus (repetition vs. non-repetition)
        seqdata,simparams = generate_seqdata_simparams("train_gonogo")
        simparams["numEpisodes"]=len(seqdata)
        loss,outputs_2,hidden_2,labels,inputs = rnn_step(simparams,seqdata) # second step forward
        #fileName = os.path.join(baseDir,'outputs','repetition_test_' + str(epoch_test) + '.p')
        fileName2 = os.path.join(baseDir,'outputs','repetition_test_' + str(epoch_test) + '.mat')
        # here save
        adict = {}
        adict['seqdata'] = seqdata
        adict['labels'] = labels.detach().numpy()
        adict['inputs'] = inputs.detach().numpy()
        adict['outputs_1'] = outputs_1.detach().numpy()
        adict['outputs_2'] = outputs_2.detach().numpy()
        adict['hidden_1'] = hidden_1.detach().numpy()
        adict['hidden_2'] = hidden_2.detach().numpy()
        scipy.io.savemat(fileName2, adict)
        # here save variables
        #pickle.dump([seqdata,labels,inputs,outputs_1,hidden_1,outputs_2,hidden_2], open(fileName, "wb" ))
        # later for opening: seqdata,labels,inputs,outputs_1,hidden_1,outputs_2,hidden_2 = pickle.load(open(fileName,"rb"))
        epoch_test +=1
    
    if epoch % 100 == 0 or loss_iter <= loss_stop:
        print('Epoch: {}........'.format(epoch), end=' ')
        print("Loss: {:.5f}".format(loss.item()))
        inputs = inputs.cpu()
        labels = labels.cpu()
        outputs = outputs.cpu()
        hidden = hidden.cpu()
        if trainparams["plotOn"]:
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
            