# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:35:29 2021

@author: Estefany Suarez
"""

import os
import numpy as np

#%% --------------------------------------------------------------------------------------------------------------------
# STATIC GLOBAL VARIABLES
# ----------------------------------------------------------------------------------------------------------------------
MODEL_ID = 2

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RES_DIR = os.path.join(PROJ_DIR, 'raw_results')
MODEL_DIR = os.path.join(RAW_RES_DIR, f'model_{MODEL_ID}')

W    = np.load(os.path.join(MODEL_DIR, 'w.npy'))
Loss = np.load(os.path.join(MODEL_DIR, 'Loss.npy'))
LE  = np.load(os.path.join(MODEL_DIR, 'Lyapunov_exp.npy'))

_, _, n_epochs = W.shape

#%% plotting
from rnns import (plotting, topology)
 
# Loss and dynamics
plotting.plot_loss((Loss[:,0],'train'), (Loss[:,1], 'test'))

plotting.plot_W(W[:,:,0])
plotting.plot_W(W[:,:,-1])

plotting.plot_LE(LE[:,:,0], 'mean')
# plotting.plot_LE(LE[:,:,1], 'std')


# Network topology    
xscaler = int(n_epochs/3) #20
yscaler = 4

strength = topology.local_topology(W[:,:,:], 'node_strength')
plotting.plot_property(strength, 'node_strength', xscaler, yscaler, plot_grad=True)

clustering = topology.local_topology(W[:,:,:], 'wei_clustering_coeff')
plotting.plot_property(clustering, 'wei_clustering_coeff', xscaler, yscaler, plot_grad=True)

centrality = topology.local_topology(W[:,:,:], 'wei_centrality')
plotting.plot_property(centrality, 'wei_centrality', xscaler, yscaler, plot_grad=True)

