# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:13:24 2021

@author: Estefany Suarez
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

from rnns import (rnns, lyapunov)

#%% Select device
# is_cuda = torch.cuda.is_available()
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")


device = torch.device("cpu")

#%% Generate task data
tau = 2
batch_size = 32
timesteps = 6400 #should be a multiple of batch_size
n_iters = int(timesteps/batch_size)

input_size = 1
train_seq = np.random.uniform(-1, 1, (timesteps+tau,input_size))
test_seq  = np.random.uniform(-1, 1, (timesteps+tau,input_size))

#%% Create training and test sequences

# training data
train_input_seq = np.stack(np.split(train_seq.copy()[tau:,:], n_iters))
train_target_seq = np.stack(np.split(train_seq.copy()[:-tau], n_iters))

train_input_seq = torch.tensor(train_input_seq, dtype=torch.float, device=device)
train_target_seq = torch.tensor(train_target_seq, dtype=torch.float, device=device)

# test data
test_input_seq = np.stack(np.split(test_seq.copy()[tau:,:], n_iters))
test_target_seq = np.stack(np.split(test_seq.copy()[:-tau], n_iters))

test_input_seq = torch.tensor(test_input_seq, dtype=torch.float, device=device)
test_target_seq = torch.tensor(test_target_seq, dtype=torch.float, device=device)


#%% Instantiate RNN model
output_size = train_target_seq.shape[-1]
hidden_size = 100

model = rnns.NeuralNet(input_size=input_size,
                       output_size=output_size,
                       hidden_size=hidden_size,
                       n_layers=1,
                       nonlinearity='tanh',
                       # init_input='ones',
                       # init_hidden='xavier_uniform',
                       # init_output='xavier_uniform',
                       bias=False,
                       device=device
                       )

model.to(device)


#%% model hyper-parameters
n_epochs = 500
lr = 0.001


# %% define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#%% model training
w = [model.rnn.all_weights[0][1].detach().to('cpu').numpy().copy()]

test_loss  = []
train_loss = []

avg_LEs = []
std_LEs = []

status = True 
for epoch in range(1, n_epochs + 1):

    # estimate Lyapunov exponents
    L, _ = lyapunov.compute_LE(train_input_seq,
                               model,
                               k=hidden_size,
                               )

    avg_LEs.append(np.mean(L.to('cpu').numpy().copy(), axis=0))
    std_LEs.append(np.std(L.to('cpu').numpy().copy(), axis=0))


    # clears existing gradients from previous epoch
    optimizer.zero_grad()

    # forward pass
    output, hidden = model(train_input_seq)
    output = output.to(device)

    # backward pass
    loss = criterion(output, train_target_seq.contiguous().view(-1, output_size))
    loss.backward() # Does backpropagation and calculates gradients

    optimizer.step() # Updates the weights accordingly

    # save weights
    w.append(model.rnn.all_weights[0][1].detach().to('cpu').numpy().copy())

    # performance
    train_output, _ = model(train_input_seq)
    train_output = train_output.to(device)

    train_r = np.abs(np.corrcoef(train_output.detach().to('cpu').numpy().squeeze(), train_target_seq.contiguous().view(-1, output_size).detach().to('cpu').numpy().squeeze())[0][1])
    train_loss.append(train_r)


    # estimate test loss
    test_output, _ = model(test_input_seq)
    test_output = test_output.to(device)

    test_r = np.abs(np.corrcoef(test_output.detach().to('cpu').numpy().squeeze(), test_target_seq.contiguous().view(-1, output_size).detach().to('cpu').numpy().squeeze())[0][1])
    test_loss.append(test_r)


    # if epoch%10 == 0:
    print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    print(f'train R = {np.round(train_r, 5)}   ...   test R = {np.round(test_r, 5)}')

    if epoch >= 3:
        if test_loss[epoch-1] < test_loss[epoch-3]:
            status = False
            break

    # if epoch%10 == 0:
    #     print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
    #     print("Loss: {:.4f}".format(loss.item()))

        # plt.scatter(output.detach().to('cpu').numpy().squeeze(), target_seq.contiguous().view(-1, output_size).detach().numpy().squeeze())
        # plt.show()


#%% concatenate data
w = np.dstack(w)

test_loss  = np.array(test_loss)
train_loss = np.array(train_loss)

avg_LEs = np.vstack(avg_LEs)
std_LEs = np.vstack(std_LEs)


#%%
from rnns import (plots)

if status:
    plots.plot_loss(train_loss)
    plots.plot_loss(test_loss)
    
    plots.plot_matrix(w[:,:,0])
    # plots.plot_matrix(w[:,:,200])
    plots.plot_matrix(w[:,:,-1])
    
    plots.plot_LE(avg_LEs, 'mean')
    plots.plot_LE(std_LEs, 'std')

#%%
from rnns import topology

strength   = topology.local_topology(w, 'node_strength')
plots.plot_property(strength)

# clustering = topology.local_topology(w, 'wei_clustering_coeff')
# plots.plot_property(clustering)

# centrality = topology.local_topology(w, 'wei_centrality')
# plots.plot_property(centrality)
