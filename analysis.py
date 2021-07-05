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

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')


def run_analysis(model_id, epoch=None):

    MODEL_DIR = os.path.join(RAW_RES_DIR, f'model_{model_id}')    
    # MODEL_DIR = os.path.join(RAW_RES_DIR, f'RND_SEED_{RND_SEED}', f'model_{model_id}')
    
    W    = np.load(os.path.join(MODEL_DIR, 'w_h.npy'))
    # W_o  = np.load(os.path.join(MODEL_DIR, 'w_h.npy'))
    Loss = np.load(os.path.join(MODEL_DIR, 'Loss.npy'))
    LE   = np.load(os.path.join(MODEL_DIR, 'Lyapunov_exp.npy'))
    
    if epoch is None: 
        _, _, epoch = W.shape
        epoch-=1


    # plotting
    from rnns import (plotting, topology)
     
    # Loss and dynamics
    plotting.plot_loss((Loss[:,0],'train'), (Loss[:,1], 'test'))
    
    # plotting.plot_W(W[:,:,0])
    # plotting.plot_W(W[:,:,50])
    # plotting.plot_W(W[:,:,100])
    # plotting.plot_W(W[:,:,-1])

    plotting.plot_LE(LE[:,:,0], 'mean')
    # plotting.plot_LE(LE[:,:,1], 'std')
    
    
    # Network topology    
    # strength = topology.local_topology(W[:,:,:epoch], 'node_strength')
    # plotting.plot_property(strength, 
    #                         plots=['value', 'grad', 'rank'], 
    #                         title='node_strength', 
    #                         scale_property=True
    #                         )

    # clustering = topology.local_topology(W[:,:,:epoch], 'wei_clustering_coeff')
    # plotting.plot_property(clustering, 
    #                         plots=['value', 'grad', 'rank'], 
    #                         title='wei_clustering_coeff', 
    #                         scale_property=True
    #                         )
    
    # centrality = topology.local_topology(W[:,:,:], 'wei_centrality')
    # plotting.plot_property(centrality, 
    #                         plots=['value', 'grad', 'rank'], 
    #                         title='wei_centrality', 
    #                         scale_property=True
    #                         )
  
