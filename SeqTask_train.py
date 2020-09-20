import torch
import matplotlib.pyplot as plt
import datetime
import os


def train(Network, trainparams, netparams, simparams, input_generator):
    # here specifications for training parameters

    # Set loss and optimizer function
    def criterion(outputs, labels, hidden, HiddenReg, DerivativeRegRate, hidden_weights):
        criterion = torch.mean((outputs - labels) ** 2) + HiddenReg * torch.sum(torch.norm(hidden, dim=-1) ** 2) + \
                    DerivativeRegRate * torch.mean(torch.norm(torch.matmul(torch.tanh(hidden), hidden_weights.T), dim=-1))
        return criterion

    optimizer = torch.optim.Adam(Network.parameters(), lr=trainparams["learningRate"], weight_decay=trainparams["L2"])

    # Train the model
    loss_iter = 1  # initialized loss
    epoch = 0      # Epoch Counter
    loss_list = []
    while loss_iter > trainparams["lossStop"] and epoch < trainparams["maxEpoch"]:
        optimizer.zero_grad()
        [inputs, labels, _, _] = input_generator(simparams)
        inputs = inputs.to(netparams["device"])
        labels = labels.to(netparams["device"])
        # forward pass
        outputs, hidden = Network(inputs, netparams["batch_size"])
        # compute the loss
        Weights = [weight for weight in Network.parameters()]
        hidden_weights = Weights[2]
        loss = criterion(outputs, labels, hidden, trainparams["HiddenRegRate"], trainparams["DerivativeRegRate"],
                         hidden_weights)
        loss_iter = loss.cpu().detach().numpy()
        loss_list.append(loss_iter)
        # backpropagation
        loss.backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0 or loss_iter <= trainparams["lossStop"]:
            print('Epoch: {}........'.format(epoch), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            inputs = inputs.cpu()
            labels = labels.cpu()
            outputs = outputs.cpu()
            hidden = hidden.cpu()
            if trainparams["plotOn"]:
                # visualize one training step
                fig, axs = plt.subplots(2, 3)
                # plt.figure(figsize=(10, 2))
                axs[0, 0].plot(inputs[:, 0, :]), axs[0, 0].set_title('Input')
                axs[0, 1].plot(labels[:, 0, :]), axs[0, 1].set_title('Target Output')
                axs[0, 2].plot(outputs.detach().numpy()[:, 0, :]), axs[0, 2].set_title('Generated Output')
                axs[1, 0].plot(hidden.detach().numpy()[:, 0, :]), axs[1, 0].set_title('Hidden States')
                axs[1, 1].plot(loss_list), axs[1, 1].set_title('MSE')
                axs[1, 2].plot(labels[:, 0, :]-outputs.detach().numpy()[:, 0, :]), axs[1, 2].set_title('Prediction Error')
                plt.show()

    # Save the trained network
    save_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    net_name = Network._get_name()
    task_name = input_generator.__name__
    if not(os.path.exists(trainparams['SaveDirectory']+'/RNN')):
        os.mkdir(trainparams['SaveDirectory'] + '/RNN')

    PATH = trainparams['SaveDirectory'] + 'RNN/' + save_time + '_' + net_name + '_' + task_name + '_' + \
           str(simparams['numTargetTrial']) + '.pth'
    torch.save(Network.state_dict(), PATH)


# This program starts training the network with a few number of targets, then gradually increases the number of
# target trials. This will speed up the training process and reduces the chance of creating a plethora region in loss.
def train_sequential(Network, trainparams, netparams, simparams, input_generator):
    # here specifications for training parameters
    # Set loss and optimizer function
    def criterion(outputs, labels, hidden, HiddenReg, DerivativeRegRate, hidden_weights):
        criterion = torch.mean((outputs - labels) ** 2) + HiddenReg * torch.sum(torch.norm(hidden, dim=-1)**2) + \
                    DerivativeRegRate* torch.mean(torch.norm(torch.matmul(torch.tanh(hidden), hidden_weights.T),dim=-1))
        return criterion

    optimizer = torch.optim.Adam(Network.parameters(), lr=trainparams["learningRate"],
                                 weight_decay=trainparams["L2"])

    # Train the model
    final_num_target = simparams['numTargetTrial']
    for num_target in range(simparams['MemorySpan'], final_num_target+1):
        print(f'Training for {num_target} output targets.')
        loss_iter = 1  # initialized loss
        epoch = 0  # Epoch Counter
        loss_list = []
        simparams['numTargetTrial'] = num_target
        while loss_iter > trainparams["lossStop"] and epoch < trainparams["maxEpoch"]:
            optimizer.zero_grad()
            [inputs, labels, _, _] = input_generator(simparams)
            inputs = inputs.to(netparams["device"])
            labels = labels.to(netparams["device"])
            # forward pass
            outputs, hidden = Network(inputs, netparams["batch_size"])
            # compute the loss
            Weights = [weight for weight in Network.parameters()]
            hidden_weights = Weights[2]
            loss = criterion(outputs, labels, hidden, trainparams["HiddenRegRate"], trainparams["DerivativeRegRate"], hidden_weights)
            loss_iter = loss.cpu().detach().numpy()
            loss_list.append(loss_iter)
            # backpropagation
            loss.backward()
            optimizer.step()
            epoch += 1
            if epoch % 100 == 0 or loss_iter <= trainparams["lossStop"]:
                print('Epoch: {}........'.format(epoch), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
                inputs = inputs.cpu()
                labels = labels.cpu()
                outputs = outputs.cpu()
                hidden = hidden.cpu()
                if trainparams["plotOn"]:
                    # visualize one training step
                    fig, axs = plt.subplots(2, 3)
                    # plt.figure(figsize=(10, 2))
                    axs[0, 0].plot(inputs[:, 0, :]), axs[0, 0].set_title('Input')
                    axs[0, 1].plot(labels[:, 0, :]), axs[0, 1].set_title('Target Output')
                    axs[0, 2].plot(outputs.detach().numpy()[:, 0, :]), axs[0, 2].set_title('Generated Output')
                    axs[1, 0].plot(hidden.detach().numpy()[:, 0, :]), axs[1, 0].set_title('Hidden States')
                    axs[1, 1].plot(loss_list), axs[1, 1].set_title('MSE')
                    axs[1, 2].plot(labels[:, 0, :] - outputs.detach().numpy()[:, 0, :]), axs[1, 2].set_title(
                        'Prediction Error')
                    plt.show()

    # Save the trained network
    save_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    net_name = Network._get_name()
    task_name = input_generator.__name__
    if not (os.path.exists(trainparams['SaveDirectory'] + '/RNN')):
        os.mkdir(trainparams['SaveDirectory'] + '/RNN')

    PATH = trainparams['SaveDirectory'] + 'RNN/' + save_time + '_' + net_name + '_' + task_name + '_' + \
           str(simparams['numTargetTrial']) + '.pth'
    torch.save(Network.state_dict(), PATH)