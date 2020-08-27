# SeqTask_main
# This is the main program for simulating sequential finger tapping task.
# Programmers: Eva + Jonathan + Mehrdad
# Date: Aug 1st, 2020

# Importing libraries
import torch
from SeqTask_taskfun import rnn_IO_gaussian, taskplot  # Loads the task function
from SeqTask_network import simple_recurrent
from SeqTask_train import train
from SeqTask_validation import test

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Felicitation! You're going to use GPU")
else:
    device = torch.device("cpu")
    print("Doh! You're using CPU")

# Fixing pseudo-random seed:
torch.manual_seed(777)

# Define Simulation Parameters
# general specifications of training simulations, trial structure
simparams = {
  "numTargets": 5,      # Number of possible targets (Pentadactyly)
  "minTarget": 0,
  "maxTarget": 5,
  "numTargetTrial": 1,  # Number of targets per trial.
  "numEpisodes": 40,    # Number of Episodes (Actually number of trials)
  "memPeriod": 10,      # Memory Period  (This will be variable in future)
  "forceWidth": 25,     # Width of force value
  "forceIPI": 10,       # Time difference between two forces
  "RT": 12,             # Reaction time (This one also will be variable in future)
  "cueOn": 8,
  "cueOff": 2,
  "preTime": 10,
  "device": device,     # Train on CPU or GPU
  "GoTrial": 0.8        # Frequency of Go trials
}

# Create and plot a sample of simulated task
rnn_input, rnn_target, _, _ = rnn_IO_gaussian(simparams)
taskplot(rnn_input, rnn_target)

# ++Define the Neural Network++
# here RNN specifications
netparams = {
    "num_classes": simparams["numTargets"],
    "input_size": rnn_input.shape[2],
    "output_size": rnn_target.shape[2],
    "hidden_size": 100,                     # number of units
    "RNN_type": "GRU",                      # Type of RNN to use "GRU" or "RNN"
    "batch_size": simparams["numEpisodes"],
    "sequence_length": rnn_input.shape[0],
    "num_layers": 1,                        # one-layer rnn
    "device": device
}

# Instantiate RNN model
rnn = simple_recurrent(netparams).to(device)
print(rnn)

trainparams = {
        "maxEpoch": 50000,     # Maximum allowed number of iterations
        "lossStop": 1e-4,      # Loss value for stopping criterion
        "plotOn": 1,           # Plot during the training process
        "L2": 0,               # L2 regularization
        "learningRate": 1e-2,  # 1e-2 for GRU, 1e-4 for RNN
        "HiddenRegRate": 1e-6,  # 1e-6 works fine for GRU
        "SaveDirectory": '.'
}

train(rnn, trainparams, netparams, simparams, input_generator=rnn_IO_gaussian)


testparams = {
        "numTrial": 10,                                 # Number of times that Task function is called
        "plotOn": 0,                                    # Plot during the training process
        "LoadDirectory": trainparams['SaveDirectory'],  # Where can it find the saved network
        "LoadFileName": '/RNN/27_08_2020_21_32_simple_recurrent_rnn_IO_gaussian_1.pth'
}

test(rnn, testparams, netparams, simparams, input_generator=rnn_IO_gaussian)

