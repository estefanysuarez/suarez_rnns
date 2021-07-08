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
from scipy.linalg import eigh

from rnns import (rnns, io_data, lyapunov)
import analysis


#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
# RND_SEED = 5

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')

DEVICE = torch.device("cpu")

#%%
def train_model(model_id, model, x):

    x_train, x_test = x

    # Forward pass
    y_train = model(x_train)
    y_test  = model(x_test)

    # np.save(os.path.join(RAW_RES_DIR, f'x_train_{model_id}.npy'), y_train.detach().numpy())
    # np.save(os.path.join(RAW_RES_DIR, f'x_test_{model_id}.npy'), y_test.detach().numpy())

    return y_train.detach().numpy(), y_test.detach().numpy()


#%%
def run_workflow(idx, alpha):

    config = configparser.ConfigParser()
    config.read('params.ini')

    # Generate IO data
    x_train, x_test = np.load('E:/P7_RC/suarez_neuromorphicnetworks2/raw_results/mem_cap/io_tasks/inputs.npy')

    x_train = x_train[np.newaxis,:,:] #.reshape(500,100,1) #
    x_test  = x_test[np.newaxis,:,:] #.reshape(500,100,1)   #

    x_train = torch.tensor(x_train, dtype=torch.float, device=DEVICE)
    x_test  = torch.tensor(x_test, dtype=torch.float, device=DEVICE)

    # io_kwargs = {'task':config['task_params']['task'],
    #              'tau': 0, #model_id, #int(config['task_params']['tau']),
    #              'seq_len':int(config['task_params']['seq_len']), #should be an integer multiple of batch_size
    #              'input_size':int(config['task_params']['input_size']),
    #              'batch_size':int(config['task_params']['batch_size']),
    #              }


    # x_train, y_train = io_data.generate_IOData(**io_kwargs)
    # x_train = torch.tensor(x_train, dtype=torch.float, device=DEVICE)
    # y_train = torch.tensor(y_train, dtype=torch.float, device=DEVICE)

    # x_test, y_test = io_data.generate_IOData(**io_kwargs)
    # x_test = torch.tensor(x_test, dtype=torch.float, device=DEVICE)
    # y_test = torch.tensor(y_test, dtype=torch.float, device=DEVICE)

    # RNN model parameters
    reservoir_params = {'input_size':x_train.shape[-1],
                        'hidden_size':int(config['model_params']['hidden_size']),
                        # 'nonlinearity':config['model_params']['nonlinearity'],
                        'device':DEVICE
                        }

    # Instantiate RNN model
    # torch.random.manual_seed(RND_SEED)
    reservoir = rnns.Reservoir(**reservoir_params)
    reservoir.to(DEVICE)

    # Connectivity data
    conn = np.load('E:/P7_RC/suarez_neuromorphicnetworks2/raw_results/conn_results/reliability/scale500/consensus_0.npy')
    conn = (conn-conn.min())/(conn.max()-conn.min())
    ew, _ = eigh(conn)
    conn  = alpha*(conn/np.max(ew))

    ctx = np.load('E:/P7_RC/suarez_neuromorphicnetworks2/data/cortical/cortical_human_500.npy')
    wi = np.zeros((len(conn), x_train.shape[-1]))
    wi[ctx == 0] = 0.0001
    
    reservoir.set_input_weights(wi)
    reservoir.set_rnn_weights(conn)

    return train_model(model_id=idx,
                       model=reservoir,
                       x=(x_train, x_test),
                       )


#%%
def three2two(x):
    return np.concatenate(x, axis=0)

def main():

    print ('\nINITIATING PROCESSING TIME', flush=True)
    t0 = time.perf_counter()

    res_train = []
    res_test  = []
    for idx, alpha in tqdm(enumerate(np.linspace(0.5,1.5,21))): #np.arange(6,11,2)[::-1]:

        # run_workflow(idx, alpha)

        y_train, y_test = run_workflow(idx, alpha)
        
        # print('\n')
        # print(y_train.shape)
        # print(y_train[:2,:3])

        y_train = three2two(y_train)
        y_test  = three2two(y_test)

        # print(y_train.shape)

        # print(y_train[:2,:3])

        res_train.append(y_train)
        res_test.append(y_test)
        
        # res_train.append(three2two(y_train))
        # res_test.append(three2two(y_test))

        # analysis.run_analysis(model_id, epoch=None)

    # reservoir_states = [(rs_train, rs_test) for rs_train, rs_test in zip(res_train, res_test)]
    # np.save('C:/Users/User/Desktop/res_states.npy', reservoir_states, allow_pickle=False)

    print ('\nTOTAL PROCESSING TIME')
    print(f'{time.perf_counter() - t0:0.4f} seconds')


if __name__ == '__main__':
    main()
