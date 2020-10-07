import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def test(Network, testparams, netparams, simparams, input_generator):
    # here specifications for training parameters

    path = testparams['LoadDirectory'] + testparams['LoadFileName']
    Network.load_state_dict(torch.load(path, map_location=simparams['device']))
    [inputs, targets, inputs_history, targets_history] = input_generator(simparams)

    # Saving network weights
    Weights = [weight.detach().numpy() for weight in Network.parameters()]
    if simparams["device"] == torch.device("cuda"):
        Weights = [weight.cpu().detach().numpy() for weight in Network.parameters()]
    Weights = np.array(Weights)

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
    np.save(save_dir_path + '/Weights', Weights)


def RDM_regression(RDMparams, simparams):
    from scipy.optimize import nnls
    stats = RDMparams['stats']
    data_unique = RDM_unique_sort(RDMparams, simparams)
    time_points = data_unique["time_points"]
    Hidden_unique = data_unique["Hidden"]
    Inputs_hist_unique = data_unique["Inputs"]
    num_condition = data_unique["num_condition"]
    # RDM Regression
    # Covariance Matrix of Hidden
    G_emp = RDM_dynamics(Hidden_unique, stats, time_points, RDMparams)
    # Covariance Matrix of Inputs
    G_mod, CAT, F = RDM_model(Inputs_hist_unique, RDMparams["features"], RDMparams["features_idx"], RDMparams)

    X = np.zeros((G_mod.shape[-1], num_condition**2))
    for i in range(G_mod.shape[-1]):
        X[i, :] = G_mod[:, :, i].reshape(-1)

    # Do nonne regression at each time and calculate FSS
    beta = np.zeros((G_mod.shape[-1], len(time_points))) # 5,time_sample
    tss = np.zeros((len(time_points), 1))
    fss = np.zeros((G_mod.shape[-1], len(time_points)))
    for i in range(len(time_points)):
        y = G_emp[:, :, i]
        y = y.reshape(-1)
        beta[:, i] = nnls(X.T, y)[0]
        # These are sums of squares of all the entries of the G - matrix
        # Here we use only diagnonal of the G - matrix: Patternvariance
        tss[i] = np.trace(G_emp[:,:, i])  # Total variance of Hiddens
        for j in range(G_mod.shape[-1]):
            fss[j, i] = np.trace(G_mod[:,:, j])*beta[j, i]

    FSS = np.sum(fss, axis=0)

    if RDMparams['Normalize']:
        fss = fss/FSS

    plt.figure(figsize=(10, 5))
    for i in range(G_mod.shape[-1]):
        plt.plot(time_points, fss[i, :], linewidth=3, linestyle=CAT['linestyle'][i])


    if RDMparams['Normalize']!=1:
        plt.plot(time_points, FSS, color=[1, 0, 0], linewidth=1, linestyle=':')
        plt.plot(time_points, tss, color=[0, 0, 0], linewidth=1, linestyle=':')
    CAT['Legend'].extend(['FSS', 'TSS'])
    plt.legend(CAT['Legend'])
    plt.ylabel('Proportion Explained variance')
    plt.xlabel('Time')

    # TEST CCA
    """
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=3)
    CCA_X = np.zeros((293, 125, 3))
    for i in range(125):
        cca.fit(Hidden_unique.transpose((1, 0, 2))[:, i, :], fss[1:].T)
        X_c, Y_c = cca.transform(Hidden_unique.transpose((1, 0, 2))[:, i, :], fss[1:].T)
        CCA_X[:, i, :] = X_c
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(125):
        time1 = 588-345
        time2 = -1

        ax.plot3D(CCA_X[:time1, i, 0], CCA_X[:time1, i, 1], CCA_X[:time1, i, 2], color='C'+str(int(Inputs_hist_unique[:, 1, 0][i])), alpha=0.4)
        ax.plot3D(CCA_X[time1:time2, i, 0], CCA_X[time1:time2, i, 1], CCA_X[time1:time2, i, 2], color='C'+str(int(Inputs_hist_unique[:, 1, 0][i])))
        break
    """
    """
    # Plot instruction cue
    cc = ['C1', 'C2']
    for cue in range(2):
        plt.axvline(x=Inputs_hist_unique[0, cue+1, 1], color=cc[cue], linestyle=':')
    # Plot go cue
    cc = ['C0', 'C1']
    for cue in range(2):
        plt.axvline(x=Inputs_hist_unique[0, cue, 3], color=cc[cue], linestyle='-.')

    # Plot execution onset
    cc = ['C0', 'C1']
    for cue in range(2):
        plt.axvline(x=Target_hist_unique[0, cue, 1], color=cc[cue], linestyle='-')
    """

def RDM_sub_space(RDMparams, simparams):
    # RDM Sub_Space
    data_unique = RDM_unique_sort(RDMparams, simparams)
    Hidden_unique_long = data_unique["Hidden_long"]
    Inputs_hist_unique = data_unique["Inputs"]
    Target_hist_unique = data_unique["Target"]

    space_dim = 3
    features = RDMparams["features"]
    features_idx = RDMparams["features_idx"]
    G_mod, CAT, F = RDM_model(Inputs_hist_unique, features, features_idx, RDMparams)

    # From go Cue of Target to the End of Execution
    time_points = range(int(Inputs_hist_unique[0, 1, 3])-10, int(Target_hist_unique[0, 1, 2]))
    stampTime = np.array([1, 1, 1]) * RDMparams["critical_time_point"]

    data = Hidden_unique_long[:, time_points, :]
    pcData = np.zeros((data.shape[0]*data.shape[1], data.shape[2]))
    for i in range(data.shape[2]):
        pcData[:, i] = data[:, :, i].reshape(-1)
    # Center the data
    pcData -= np.mean(pcData, axis=0)

    # Project data on desired spaces
    V = []
    L = []
    fig = plt.figure(figsize=(15, 5))
    for i in range(len(features)):
        if RDMparams["center"]:
            F[i] -= np.mean(F[i], axis=0)
        # Projection Matrix
        P = F[i] @ np.linalg.pinv(F[i])
        Y = np.squeeze(Hidden_unique_long[:, stampTime[i], :])
        l, v = np.linalg.eig(Y.T@P.T@P@Y)
        #l, v = np.linalg.eig(Y.T @ F[i] @ F[i].T @ Y)
        idx = np.flip(np.argsort(np.real(l)))
        l = np.real(l[idx])
        v = np.real(v[:, idx])
        score = pcData@v[:, :space_dim]
        V.append(v)
        L.append(l)
        pc = np.reshape(score.T, (space_dim, len(time_points), 125), order='F')

        ax = fig.add_subplot(131 + i, projection='3d')
        for cond in range(0, 125, 2):  # range(pc.shape[-1]):
            ax.plot3D(pc[0, :, cond], pc[1, :, cond], pc[2, :, cond], color='C'+str(int(Inputs_hist_unique[:, i+1, 0][cond])))
            go_cue_time = 0
            press_onset = np.where(np.array(time_points)== RDMparams["critical_time_point"])
            end_time = -1
            ax.scatter3D(pc[0, go_cue_time, cond], pc[1, go_cue_time, cond], pc[2, go_cue_time, cond], marker='d',
                         color='C'+str(int(Inputs_hist_unique[:, i+1, 0][cond])), label='Go Cue')

            ax.scatter3D(pc[0, press_onset, cond], pc[1, press_onset, cond], pc[2, press_onset, cond], marker='o',
                         color= 'C' + str(int(Inputs_hist_unique[:, i + 1, 0][cond])), label='Press onset')

            ax.scatter3D(pc[0, end_time, cond], pc[1, end_time, cond], pc[2, end_time, cond], marker='x',
                         color='C' + str(int(Inputs_hist_unique[:, i + 1, 0][cond])), label='End')
        ax.set_title(features[i])

    basis_image = np.zeros((len(V), len(V)))
    for i in range(len(V)):
        for j in range(len(V)):
            C = V[i][:, :space_dim].T@V[j][:, :space_dim]
            basis_image[i, j] = np.sum(np.abs(C))


    print(basis_image)
    plt.figure()
    plt.bar([1, 2, 3], [basis_image[0, 1], basis_image[0, 2], basis_image[1, 2]])
    plt.xticks([1, 2, 3], ['+0 to +1', '+0 to + 2', '+1 to +2'])
    plt.ylabel('Sum of abs of inner product of bases')

    # Data Projected on subspaces
    data_subspace = []
    for i in range(len(features_idx)):
        data_subspace.append(data @ V[i])
    # Concatenate all (conditions, time samples, and neurons) for each (3) sub spaces
    data_subspace_flatten = np.zeros((len(data_subspace[0].reshape(-1)), len(features_idx)))
    for i in range(len(features_idx)):
        data_subspace_flatten[:,i] = data_subspace[i].reshape(-1)

    orthogonality = np.abs(corrcov(data_subspace_flatten.T @ data_subspace_flatten))
    print(orthogonality)
    plt.figure()
    plt.bar([1, 2, 3], [orthogonality[0, 1], orthogonality[0, 2], orthogonality[1, 2]])
    plt.xticks([1, 2, 3], ['+0 to +1', '+0 to + 2', '+1 to +2'])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('Orthogonality of Feature Activation')

    return data, V, L



# RDM_unique_sort
# Finds sorted unique condition for target trial. Example: Target, Next, Next 2
def RDM_unique_sort(RDMparams, simparams):
    # Load network input, output, and hidden states
    Hiddens = np.load(RDMparams['LoadDirectory'] + '/Hiddens.npy')
    Inputs_history = np.load(RDMparams['LoadDirectory'] + '/Inputs_history.npy')
    Targets_history = np.load(RDMparams['LoadDirectory'] + '/Targets_history.npy')

    # Loading input and target values for plotting alongside RDM regression
    Inputs = np.load(RDMparams['LoadDirectory'] + '/Inputs.npy')
    Outputs = np.load(RDMparams['LoadDirectory'] + '/Outputs.npy')

    # Number of possible condition: Eg: 5 fingers, 3 memory ---> 125 conditions
    num_condition = simparams["numTargets"] ** simparams["MemorySpan"]
    # Which target in the episode to get as current targte (+0)
    which_target = RDMparams['which_target']

    # Possible analysis time points
    # The whole target: instruction of target till end of execution of target
    ## time_points = range(int(Inputs_history[0, which_target, 1]-10), int(Targets_history[0, which_target, -1]+20))
    # Only the execution of current target
    time_points = range(int(Inputs_history[0, which_target, 3]-10), int(Inputs_history[0, which_target+3, 1]))
    # From Current Target's instruction cue till right before the first instruction after execution of current target
    ## time_points = range(int(Inputs_history[0, which_target, 1]-10), int(Inputs_history[0, which_target+3, 1]))

    # Sort based on current Target, Next Target, Previous Target
    ## history_slice = Inputs_history[:, [which_target, which_target + 1, which_target - 1], 0]
    # Sort based on current Previous Target, Target, Next Target
    ## history_slice = Inputs_history[:, [which_target-1, which_target, which_target + 1], 0]
    # Sort based on current Target, Next Target, Next +
    history_slice = Inputs_history[:, [which_target, which_target + 1, which_target + 2], 0]
    uniques, index, count = np.unique(history_slice, axis=0, return_inverse=True, return_counts=True)

    if len(count) != num_condition:
        print("Not all possible combinations exist in Data!")

    data_unique = {}
    min_num_condition = min(count)   # Minimum number of repeated condition
    data_unique["Inputs"] = np.zeros((min_num_condition, num_condition, 4, 5))
    data_unique["Target"] = np.zeros((min_num_condition, num_condition, 4, 3))
    data_unique["Hidden"] = np.zeros((min_num_condition, num_condition, len(time_points), Hiddens.shape[2])) # Rep, Cond, time, Unit
    data_unique["Hidden_long"] = np.zeros((min_num_condition, num_condition, Hiddens.shape[0], Hiddens.shape[2]))
    for cond in range(len(count)):
        data_unique["Inputs"][:, cond, :] = Inputs_history[np.where(index == cond)[0][:min(count)],
                                         which_target - 1:which_target + 3, :]
        data_unique["Target"][:, cond, :] = Targets_history[np.where(index == cond)[0][:min(count)],
                                         which_target - 1:which_target + 3, :]
        temp = Hiddens[:, np.where(index == cond)[0][:min(count)], :].transpose((1, 0, 2))
        data_unique["Hidden_long"][:, cond, :, :] = temp # Unique with original length
        data_unique["Hidden"][:, cond, :, :] = temp[:, time_points, :] # Unique with length equal to time_points

    # Mean over similar trials/ or just get the first one.
    data_unique["num_condition"] = num_condition
    data_unique["time_points"] = time_points
    data_unique["Hidden"] = np.mean(data_unique["Hidden"], axis=0) # Mean of activity over similar trials
    data_unique["Hidden_long"] = np.squeeze(data_unique["Hidden_long"][0])  # Here just got the first
    data_unique["Inputs"] = np.squeeze(data_unique["Inputs"][0])    # Here just got the first
    data_unique["Target"] = np.squeeze(data_unique["Target"][0])    # Here just got the first
    data_unique["Input_net"] = Inputs
    data_unique["Output_net"] = Outputs
    return data_unique

##################
# RDM functions
def indicatorMatrix(what,c):
    try:
        row, col = c.shape
    except ValueError:
        c = np.expand_dims(c,axis=-1)
        row, col = c.shape

    transp = 0
    if row == 1:
        c = c.T
        transp = 1

    a, cc = np.unique(c,return_inverse=True, axis=0) # Make the class -labels 1-K
    K = max(cc)+1 # number of classes, assuming numbering from 1...max(c)

    if what == 'identity':
        # Dummy coding matrix
        X = np.zeros((row, K))
        for i in range(K):
            X[cc == i, i] = 1

    if what == 'allpairs':   # all possible pairs
        X = np.zeros((row, int(K * (K - 1) / 2)))
        k = 0
        for i in range(K):
            for j in range(i+1, K):
                X[cc == i, k] = 1. / sum(cc == i)
                X[cc == j, k] = -1. / sum(cc == j)
                k = k + 1
    # Transpose design matrix
    if transp == 1:
        X = X.T
    else:
        a = a.T
    return X, a


# RDM_dynamics
def RDM_dynamics(Hidden_unique, stats, time_points, RDMparams):
    K = Hidden_unique.shape[0]
    H = np.eye(K) - np.ones((K, K)) / K
    # C, _ = indicatorMatrix('allpairs', np.arange(0,K).T)
    G_emp = np.zeros((K, K, len(time_points)))
    # Generate the plot
    for t in range(Hidden_unique.shape[1]):
        Zslice = Hidden_unique[:, t, :]
        if stats == 'D':
            pass # Will be filled later
           # Diff = C * Zslice
           # G_emp(:,:, i)=squareform(sum(Diff. * Diff, 2))
        if stats == 'G':
            G_emp[:, :, t] = H @ Zslice @ Zslice.T @ H.T

    # Plot the RDM Matrices
    if RDMparams['plotOn'] == 1:
        num_plots = np.minimum(G_emp.shape[-1], 5)
        fig, axs = plt.subplots(1, 5)
        for sub in range(num_plots):
            axs[sub].imshow(G_emp[:, :, sub+10])
            axs[sub].set_title('G: t= '+str(sub+10))

    return G_emp

def RDM_model(Inputs_hist_unique, features, features_idx, RDMparams):
    stats = RDMparams['stats']
    CAT = {}
    CAT['Color'] = []
    CAT['linestyle'] = ['-', '-', '-', '-', '-', '-', '-']
    CAT['Legend'] = []
    K = Inputs_hist_unique.shape[0]
    H = np.eye(K) - np.ones((K, K)) / K
    # C, _ = indicatorMatrix('allpairs', np.arange(0,K).T)
    #features = ['target', 'prev', 'next', 'prevT', 'nextT'] # Combination of 'fingers','transitions','sequence'
    #features_idx = [1, 0, 2, [0, 2], [1, 2]]
    G_mod = np.zeros((K, K, len(features)))
    F = []
    for i, feat in enumerate(features, 0):
        Z, _ = indicatorMatrix('identity', Inputs_hist_unique[:, features_idx[i], 0])
        CAT['Color'].append('C' + str(i))
        CAT['Legend'].append(feat)
        F.append(Z)
        if stats == 'D':
            # for later
            # Diff = C * Z{i}
            # G_mod(:,:, i)=squareform(sum(Diff. * Diff, 2))
            pass
        if stats == 'G':
            G_mod[:, :, i]= H @ Z @ Z.T @ H.T
            # Normalize the variance
            G_mod[:, :, i] = G_mod[:, :, i] / np.trace(G_mod[:, :, i])

    # Plot the RDM Matrices
    if RDMparams['plotOn'] == 1:
        fig, axs = plt.subplots(1, len(features))
        for sub in range(G_mod.shape[-1]):
            axs[sub].imshow(G_mod[:, :, sub])
            axs[sub].set_title(CAT['Legend'][sub])
        plt.show()
    return G_mod, CAT, F

def corrcov(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation