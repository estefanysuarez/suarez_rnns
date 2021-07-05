# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:44:16 2021

@author: Estefany Suarez
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"

import configparser
import time
from tqdm import tqdm 

import torch
from torch import nn
from torch.optim.lr_scheduler import (StepLR, LambdaLR, MultiplicativeLR, ExponentialLR)

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig

from rnns import (rnns, io_data, lyapunov)
import analysis


#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
RND_SEED = 5

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

DEVICE = torch.device("cpu")

#%%
def train_model(model_id, model, x, y, n_epochs, lr, gamma, start_decay, sparsity=None):
    
    RES_DIR = os.path.join(RAW_RES_DIR, f'model_{model_id}')
    if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)
        
    print ('\nINITIATING PROCESSING TIME', flush=True)
    t0 = time.perf_counter()
    
    x_train, x_test = x
    y_train, y_test = y

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    if sparsity is not None: model.apply_sparse_structure(sparsity)
    w_h = [model.get_rnn_weights(1)]
    w_o = [model.get_output_weights()]
    
    test_loss  = []
    train_loss = []
    
    avg_LE = []
    std_LE = []
    
    status = True 
    for epoch in tqdm(range(1, n_epochs + 1)):

        # Estimate Lyapunov exponents
        LE, _ = lyapunov.compute_LE(x_train,
                                    model,
                                    k=model.hidden_size,
                                    warmup=0,
                                    T_ons=1
                                    )
    
        stats_LE = torch.std_mean(LE, axis=0)
        std_LE.append(stats_LE[0])
        avg_LE.append(stats_LE[1])
    

        # Clear existing gradients from previous epoch
        optimizer.zero_grad() 

        # Forward pass
        output, hidden = model(x_train)
        output = output.to(DEVICE)
        
        # Backward pass
        loss = criterion(output, y_train.contiguous().view(-1, model.output_size))
        loss.backward() # Does backpropagation and calculates gradients

        optimizer.step() # Updates the weights accordingly
        if epoch > start_decay: scheduler.step() # Updates the learning rate    
        

        # Save weights
        if sparsity is not None: model.apply_sparse_structure(sparsity)
        w_h.append(model.get_rnn_weights(1))
        w_o.append(model.get_output_weights())

        # Performance - training set
        train_output, _ = model(x_train)
        train_output = train_output.to(DEVICE)
        train_r = np.abs(np.corrcoef(train_output.detach().to('cpu').numpy().squeeze(), y_train.contiguous().view(-1, model.output_size).detach().to('cpu').numpy().squeeze())[0][1])
        train_loss.append(train_r)

        # Performance - test set
        test_output, _ = model(x_test)
        test_output = test_output.to(DEVICE)
        test_r = np.abs(np.corrcoef(test_output.detach().to('cpu').numpy().squeeze(), y_test.contiguous().view(-1, model.output_size).detach().to('cpu').numpy().squeeze())[0][1])
        test_loss.append(test_r)
    
        if epoch%10 == 0:
            # print('\nEpoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            # print("Loss: {:.4f}".format(loss.item()))
            print(f'Epoch: {epoch}/{n_epochs} ........', end=' ')
            print(f'R = {np.round(train_r, 5)} - Rval = {np.round(test_r, 5)}')


    print ('\nTOTAL PROCESSING TIME')
    print(f'{time.perf_counter() - t0:0.4f} seconds')

    np.save(os.path.join(RES_DIR, 'w_o.npy'), np.dstack(w_o))        
    np.save(os.path.join(RES_DIR, 'w_h.npy'), np.dstack(w_h))
    np.save(os.path.join(RES_DIR, 'loss.npy'), np.column_stack((train_loss,test_loss)))
    np.save(os.path.join(RES_DIR, 'lyapunov_exp.npy'), np.dstack((np.vstack(avg_LE),np.vstack(std_LE))))
    
    return status

#%%
def run_workflow(model_id):
    
    config = configparser.ConfigParser()
    config.read('params.ini')

    # Generate IO data
    io_kwargs = {'task':config['task_params']['task'],
                 'tau': model_id, #int(config['task_params']['tau']),
                 'seq_len':int(config['task_params']['seq_len']), #should be an integer multiple of batch_size
                 'input_size':int(config['task_params']['input_size']),
                 'batch_size':int(config['task_params']['batch_size']),
                 }
    
    x_train, y_train = io_data.generate_IOData(**io_kwargs)
    x_train = torch.tensor(x_train, dtype=torch.float, device=DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float, device=DEVICE)
    
    x_test, y_test = io_data.generate_IOData(**io_kwargs)
    x_test = torch.tensor(x_test, dtype=torch.float, device=DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float, device=DEVICE)


    # RNN model parameters  
    rnn_params = {'input_size':x_train.shape[-1],
                  'output_size':y_train.shape[-1],
                  'hidden_size':int(config['model_params']['hidden_size']),
                  'n_layers':int(config['model_params']['n_layers']),
                  'nonlinearity':config['model_params']['nonlinearity'],
                  'init_input':eval(config['model_params']['init_input']),
                  'init_hidden':eval(config['model_params']['init_hidden']),
                  'init_output':eval(config['model_params']['init_output']),
                  'bias':bool(config['model_params']['bias']),
                  'device':DEVICE
                  }
    
    # Instantiate RNN model
    torch.random.manual_seed(RND_SEED)
    model = rnns.NeuralNet(**rnn_params)                 
    model.to(DEVICE)
    
    
    # Sparse model parameters  
    G = nx.connected_watts_strogatz_graph(n=rnn_params['hidden_size'], 
                                          k=5, p=0.5, 
                                          tries=100, 
                                          seed=RND_SEED
                                          )
    
    # Apply sparse mask
    sparse_w = nx.to_numpy_array(G)
    model.apply_sparse_structure(sparse_w, update=False)

    
    # Loss, Optimizer, and model Hyper-parameters
    return train_model(model_id=model_id,
                       model=model, 
                       x=(x_train, x_test), 
                       y=(y_train, y_test), 
                       n_epochs=int(config['train_params']['n_epochs']), 
                       lr=float(config['train_params']['lr']), 
                       gamma=float(config['train_params']['gamma']), 
                       start_decay=int(config['train_params']['start_decay']),
                        # sparsity=sparse_w,
                       )


#%%
def main():
    
    for model_id in np.arange(2,17,2)[::-1]: 
            
        print(f'\n--------------------------------- TAU = {model_id}')
        
        status = run_workflow(model_id)
        print(f'Model No. {model_id} - status: {status}')
    
        analysis.run_analysis(model_id, epoch=None)


if __name__ == '__main__': 
    main()