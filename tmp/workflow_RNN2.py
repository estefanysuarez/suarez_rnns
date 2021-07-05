# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:13:24 2021

@author: Estefany Suarez
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import configparser
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import (StepLR, LambdaLR, MultiplicativeLR, ExponentialLR)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

from rnns import (rnns, io_data, lyapunov)

#%% Select device
# is_cuda = torch.cuda.is_available()
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

device = torch.device("cpu")

#%% Generate IO data for task
config = configparser.ConfigParser()
config.read('params_1.ini')

io_kwargs = {'task':config['task_params']['task'],
             'tau':int(config['task_params']['tau']),
             'seq_len':int(config['task_params']['seq_len']), #should be an integer multiple of batch_size
             'input_size':int(config['task_params']['input_size']),
             'batch_size':int(config['task_params']['batch_size']),
             }

x_train, y_train = io_data.generate_IOData(**io_kwargs)
x_train = torch.tensor(x_train, dtype=torch.float, device=device)
y_train = torch.tensor(y_train, dtype=torch.float, device=device)

x_test, y_test = io_data.generate_IOData(**io_kwargs)
x_test = torch.tensor(x_test, dtype=torch.float, device=device)
y_test = torch.tensor(y_test, dtype=torch.float, device=device)


#%% Instantiate RNN model
rnn_params = {'input_size':x_train.shape[-1],
              'output_size':y_train.shape[-1],
              'hidden_size':int(config['model_params']['hidden_size']),
              'n_layers':int(config['model_params']['n_layers']),
              'nonlinearity':config['model_params']['nonlinearity'],
              'init_input':eval(config['model_params']['init_input']),
              'init_hidden':eval(config['model_params']['init_hidden']),
              'init_output':eval(config['model_params']['init_output']),
              'bias':bool(config['model_params']['bias']),
              'device':device
              }

# print(f'N. NEURONS : {model_params['hidden_size']}', flush=True)
model = rnns.NeuralNet(**rnn_params)                 
model.to(device)

# %% define Loss, Optimizer, and model Hyper-parameters
n_epochs = int(config['train_params']['n_epochs'])
lr = float(config['train_params']['lr'])
gamma = float(config['train_params']['gamma'])
start_decay = int(config['train_params']['start_decay'])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=gamma)

#%% model training
w = [model.rnn.all_weights[0][1].detach().numpy().copy()]

test_loss  = []
train_loss = []

avg_LE = []
std_LE = []

status = True 
print ('\nINITIATING PROCESSING TIME', flush=True)
t0 = time.perf_counter()

for epoch in range(1, n_epochs + 1):

    # print('\nEpoch: {}/{}.............'.format(epoch, n_epochs), flush=True)

    # Estimate Lyapunov exponents
    # print('\tEstimating Lyapunov exponents ...', flush=True)
    # tic = time.perf_counter()
    LE, _ = lyapunov.compute_LE(x_train,
                                model,
                                k=rnn_params['hidden_size'],
                                warmup=0,
                                T_ons=1
                                )
    
    # print(f'\n\t\tProcessing time for LEs: {time.perf_counter()-tic:0.4f}', flush=True)

    stats_LE = torch.std_mean(LE, axis=0)
    std_LE.append(stats_LE[0])
    avg_LE.append(stats_LE[1])
    
    # Train RNN 
    # print('\n\tTraining RNN ...', flush=True)
    # tic = time.perf_counter()

    # Clear existing gradients from previous epoch
    optimizer.zero_grad() 

    # Forward pass
    output, hidden = model(x_train)
    output = output.to(device)

    # Backward pass
    loss = criterion(output, y_train.contiguous().view(-1, rnn_params['output_size']))
    loss.backward() # Does backpropagation and calculates gradients

    optimizer.step() # Updates the weights accordingly
    if epoch > start_decay: scheduler.step() # Updates the learning rate    

    # print(f'\t\tProcessing time for training: {time.perf_counter()-tic:0.4f}', flush=True)

    # Save weights
    w.append(model.rnn.all_weights[0][1].detach().numpy().copy())


    # Performance - training set
    # print('\n\tEstimating training performance')
    # tic = time.perf_counter()
    train_output, _ = model(x_train)
    train_output = train_output.to(device)
    train_r = np.abs(np.corrcoef(train_output.detach().to('cpu').numpy().squeeze(), y_train.contiguous().view(-1, rnn_params['output_size']).detach().to('cpu').numpy().squeeze())[0][1])
    train_loss.append(train_r)
    # print(f'\t\tProcessing time training performance: {time.perf_counter()-tic:0.4f}')

    # Performance - test set
    # print('\n\tEstimating test performance')
    # tic = time.perf_counter()
    test_output, _ = model(x_test)
    test_output = test_output.to(device)
    test_r = np.abs(np.corrcoef(test_output.detach().to('cpu').numpy().squeeze(), y_test.contiguous().view(-1, rnn_params['output_size']).detach().to('cpu').numpy().squeeze())[0][1])
    test_loss.append(test_r)
    # print(f'\t\tProcessing time test performance: {time.perf_counter()-tic:0.4f}')

    # Early stopping
    if test_loss[epoch-1] < test_loss[epoch-2]:
        status = False
        break
    
    if epoch%10 == 0:
        print('\nEpoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        print(f'Training R = {np.round(train_r, 5)}   ...   Test R = {np.round(test_r, 5)}')


print ('\nTOTAL PROCESSING TIME')
print(f'{time.perf_counter() - t0:0.4f} seconds')


#%% concatenate data
# w = np.dstack(w)

# test_loss  = np.array(test_loss)
# train_loss = np.array(train_loss)

# avg_LE = np.vstack(avg_LE)
# std_LE = np.vstack(std_LE)


