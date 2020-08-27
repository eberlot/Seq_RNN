import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def test(Network, testparams, netparams, simparams, input_generator):
    # here specifications for training parameters

    path = testparams['LoadDirectory'] + testparams['LoadFileName']
    Network.load_state_dict(torch.load(path))
    [inputs, targets, inputs_history, targets_history] = input_generator(simparams)

    Inputs = np.zeros((inputs.size(0), testparams['numTrial']*simparams['numEpisodes'], inputs.size(2)))
    Targets = np.zeros((targets.size(0), testparams['numTrial']*simparams['numEpisodes'], targets.size(2)))
    Outputs = np.zeros((targets.size(0), testparams['numTrial']*simparams['numEpisodes'], targets.size(2)))
    Hiddens = np.zeros((inputs.size(0), testparams['numTrial'] * simparams['numEpisodes'], netparams['hidden_size']))
    Inputs_history = np.zeros((testparams['numTrial'] * simparams['numEpisodes'], inputs_history.shape[1], inputs_history.shape[2]))
    Targets_history = np.zeros((testparams['numTrial'] * simparams['numEpisodes'], targets_history.shape[1], targets_history.shape[2]))
    with torch.no_grad():
        for i in range(testparams['numTrial']):
            [inputs, targets, inputs_history, targets_history] = input_generator(simparams)
            inputs = inputs.to(netparams["device"])
            targets = targets.to(netparams["device"])
            # forward pass
            outputs, hidden = Network(inputs, netparams["batch_size"])

            inputs = inputs.cpu()
            targets = targets.cpu()
            outputs = outputs.cpu()
            hidden = hidden.cpu()

            Inputs[:, i*simparams['numEpisodes']:(i+1)*simparams['numEpisodes'], :] = inputs.numpy()
            Targets[:, i*simparams['numEpisodes']:(i+1)*simparams['numEpisodes'], :] = targets.numpy()
            Hiddens[:, i * simparams['numEpisodes']:(i+1) * simparams['numEpisodes'], :] = hidden.numpy()
            Outputs[:, i * simparams['numEpisodes']:(i+1) * simparams['numEpisodes'], :] = outputs.numpy()
            Inputs_history[i * simparams['numEpisodes']:(i + 1) * simparams['numEpisodes'], :, :] = inputs_history
            Targets_history[i*simparams['numEpisodes']:(i+1)*simparams['numEpisodes'], :, :] = targets_history
            if testparams["plotOn"]:
                # visualize one training step
                fig, axs = plt.subplots(2, 3)
                # plt.figure(figsize=(10, 2))
                axs[0, 0].plot(inputs[:, 0, :]), axs[0, 0].set_title('Input')
                axs[0, 1].plot(targets[:, 0, :]), axs[0, 1].set_title('Target Output')
                axs[0, 2].plot(outputs.detach().numpy()[:, 0, :]), axs[0, 2].set_title('Generated Output')
                axs[1, 0].plot(hidden.detach().numpy()[:, 0, :]), axs[1, 0].set_title('Hidden States')
                axs[1, 1].plot(targets[:, 0, :]-outputs.detach().numpy()[:, 0, :]), axs[1, 2].set_title('Prediction Error')
                plt.show()

    save_dir_path = testparams['LoadDirectory']+testparams['LoadFileName'][:-4]
    if not(os.path.exists(save_dir_path)):
        os.mkdir(save_dir_path)

    np.save(save_dir_path + '/Inputs', Inputs)
    np.save(save_dir_path + '/Targets', Targets)
    np.save(save_dir_path + '/Outputs', Outputs)
    np.save(save_dir_path + '/Hiddens', Hiddens)
    np.save(save_dir_path + '/Inputs_history', Inputs_history)
    np.save(save_dir_path + '/Targets_history', Targets_history)