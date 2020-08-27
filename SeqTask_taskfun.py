# SeqTask_taskfun
# This module contains various functions for creating input-output values of
# a Sequential task under different conditions.
# Programmers: Eva + Jonathan + Mehrdad
# Date: Aug 1st, 2020

# Importing Libraries
import numpy as np
import torch
import random


# Simple Gaussian force
# Task Structure: Pre->S1->S2...Sn->Mem->RT->E1->E2...En
# Force profile: crude gaussian force profile
def rnn_IO_gaussian(simparams):
    # Update simparams:
    # Instruction time: (Cue-on + Cue-off) for all targets + Memory period
    simparams.update({"instTime": (simparams["cueOn"] + simparams["cueOff"]) * simparams["numTargetTrial"] +
                                  +simparams["cueOn"] + simparams["memPeriod"]})
    # Movement time: Force IPI for all targets + Force width
    simparams.update({"moveTime": (simparams["forceIPI"] * simparams["numTargets"]) + simparams["forceWidth"]})
    # Total Trial Time: Instruction + Reaction + Movement
    simparams.update({"trialTime": simparams["instTime"] + simparams["RT"] + simparams["moveTime"]})

    trial_n = simparams["numEpisodes"]
    in_data = np.zeros([simparams["trialTime"], simparams["numEpisodes"], simparams["numTargets"]+1])
    out_data = np.zeros([simparams["trialTime"], trial_n, simparams["numTargets"]])
    GoTrial = random.choices([0, 1], weights=[1-simparams["GoTrial"], simparams["GoTrial"]], k=simparams["numEpisodes"])
    y = gaussian()
    # Zero matrices for task info
    # inputs_history: A zero matrix for recording task's input timing
    # NumEpisodes, NumTargets + 1 GoSignal, 3(label, start-time, end_time)
    inputs_history = np.zeros((trial_n, simparams["numTargetTrial"]+1, 3))
    inputs_history[:, -1, 0] = GoTrial
    targets_history = np.zeros((trial_n, simparams["numTargetTrial"], 3))
    for i in range(trial_n):
        seq_data = np.random.randint(simparams["minTarget"], high=simparams["maxTarget"], size=[1, simparams["numTargetTrial"]])
        inputs_history[i, :-1, 0] = seq_data  # Saving labels for the current finger sequence for the current trial
        targets_history[i, :, 0] = seq_data
        t = simparams["preTime"]
        for j in range(simparams["numTargetTrial"]): # define targets
            inputs_history[i, j, 1:] = [t, t+simparams["cueOn"]]   # Saving start and end time for each instruction
            t_inp = range(t, t+simparams["cueOn"])
            in_data[t_inp, i, int(seq_data[0, j])] = 1
            t = t + simparams["cueOn"]+simparams["cueOff"]
        # whether go or no-go trial
        if GoTrial[i] == 1:  # go trial
            in_data[range(t+simparams["memPeriod"], t+simparams["memPeriod"] +
                      simparams["cueOn"]), i, simparams["numTargets"]] = 1  # go signal
            # Saving GoCue signal interval
            inputs_history[i, -1, 1:] = [t+simparams["memPeriod"], t+simparams["memPeriod"]+simparams["cueOn"]]
            # expected output
            t = simparams["instTime"]+simparams["RT"]
            for j in range(simparams["numTargetTrial"]):
                t_out = range(t, t+simparams["forceWidth"])
                targets_history[i, j, 1:] = [t, t+simparams["forceWidth"]]
                previous = out_data[t_out, i, int(seq_data[0, j])]
                target = y
                out_data[t_out, i, int(seq_data[0, j])] = np.maximum(previous, target)
                t = t + simparams["forceIPI"]
    inputs = torch.from_numpy(in_data)
    target_outputs = torch.from_numpy(out_data)
    return inputs.float().to(simparams["device"]), target_outputs.float().to(simparams["device"]), inputs_history, targets_history

# convolve expected output force profile with a Gaussian window - for now hard-coded


def gaussian():
    x = np.arange(-12.5, 12.5, 1)
    s = 3
    y = 1./np.sqrt(2.*np.pi*s**2) * np.exp(-x**2/(2.*s**2))
    y = y/np.max(y)
    return y


def taskplot(rnn_input, rnn_target):
    import matplotlib.pyplot as plt
    inputs = rnn_input.cpu().numpy()
    targets = rnn_target.cpu().numpy()
    rand_trial_inx = np.random.randint(0, inputs.shape[1], 3)

    fig, axs = plt.subplots(2, 3)
    for i, inx in enumerate(rand_trial_inx, 0):
        axs[0, i].plot(inputs[:, inx, :] * np.array([1, 2, 3, 4, 5, 6]))
        axs[0, i].set_title(f'Trial #{inx+1}\nInput')
        axs[0, i].legend(['Finger 1', 'Finger 2', 'Finger 3', 'Finger 4', 'Finger 5', 'Go'])
        axs[0, i].set_ylim([0, 6.2])
        axs[1, i].plot(targets[:, inx, :])
    plt.show()



