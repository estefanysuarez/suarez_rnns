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

from rnns import (rnns, connectivity)


#%% generate task data
tau = 2
batch_size = 10 #32
timesteps = 5000 #6400
rand_var = np.random.uniform(-1,1,timesteps+tau)

input_size = 10 
input_seq = np.repeat(rand_var[:,np.newaxis], input_size, axis=1)[tau:,:]
input_seq = np.stack(np.split(input_seq, batch_size))

target_seq = rand_var[:-tau]
target_seq = np.stack(np.split(target_seq, batch_size))


#%% select device
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#%% load connectivity matrix
w = np.load('F:/Data/Lausanne/Lausanne_npy_ind/struct/struct_num_scale033.npy').mean(axis=2).astype(int) #load connectivity matrix
mask = np.loadtxt('F:/Data/Lausanne/Lausanne_txt_avg/struct_num/bin/weights033.txt').astype(int) #load binary mask
np.fill_diagonal(mask, 1)
w = w*mask # apply mask to select only consistent structural connections


#%%
# upscale connectivity matrix
density = 2*np.ones(len(w), dtype=int)
# density = np.random.randint(1,10,len(conn)) 

neuron_ids_per_roi, scld_w, scld_mask = connectivity.upscale_rnn(original_network=w, 
                                                                 fraction_intrinsic=None,
                                                                 neuron_density=density,
                                                                 intrinsic_sparsity=0.8, 
                                                                 extrinsic_sparsity=0.2,
                                                                 allow_self_conns=True,
                                                                 return_mask=True,
                                                                 )

init_scld_mask = scld_w.copy().astype(bool).astype(int)

#%% get model dimensions
output_size = 1
hidden_size = len(scld_w)


#%% input connectivity matrix
# win = np.random.randint(0, 2, (hidden_size, input_size))
# win = np.ones((hidden_size, input_size), dtype=int)


#%% instantiate BioRNN
model = rnns.BioNeuralNet(input_size=input_size,
                          output_size=output_size,
                          hidden_size=hidden_size,
                          win=None, #win,
                          w=scld_w,
                          remap_w=True,
                          init_method='xavier_uniform',
                          device=device
                          )

model.to(device)


#%% model hyper-parameters
n_epochs = 500
lr = 0.01


# %% define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#%% model training
input_seq = torch.tensor(input_seq, dtype=torch.float)
target_seq = torch.tensor(target_seq, dtype=torch.float)

track_w = [model.get_rnn_weights().detach().numpy().copy()]
track_loss = []
for epoch in range(1, n_epochs + 1):

    optimizer.zero_grad() # Clears existing gradients from previous epoch
    input_seq = input_seq.to(device)

    output, hidden = model(input_seq)

    output = output.to(device)
    target_seq = target_seq.to(device)

    loss = criterion(output, target_seq.view(-1)) #.long()
    loss.backward() # Does backpropagation and calculates gradients

    optimizer.step() # Updates the weights accordingly

    # model.set_rnn_weights(model.get_rnn_weights().detach().numpy().copy()*scld_mask)
    model.set_rnn_weights(model.get_rnn_weights().detach().numpy().copy()*init_scld_mask)

    track_w.append(model.get_rnn_weights().detach().numpy().copy())
    track_loss.append(np.abs(np.corrcoef(output.detach().numpy().squeeze(), rand_var[:-tau])[0][1]))

    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        print(np.abs(np.corrcoef(output.detach().numpy().squeeze(), rand_var[:-tau])[0][1]))
        plt.scatter(output.detach().numpy().squeeze(), rand_var[:-tau])
        plt.show()



#%%
# weights_evol = np.dstack(track_w)
# np.save('C:/Users/User/Desktop/weights.npy', weights_evol)
# np.save('C:/Users/User/Desktop/neuron_ids.npy', neuron_ids_per_roi)


#%%
sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(num=1, figsize=(13,10))
ax = plt.subplot(111)

mapp = plt.imshow(scld_w, 
                  # vmin=0, vmax=1, 
                  cmap='viridis', #'Greys',  
                  # interpolation='nearest'
                  )

fig.colorbar(mapp, ax=ax)
plt.show()
plt.close()


sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(num=2, figsize=(13,10))
ax = plt.subplot(111)

mapp = plt.imshow(track_w[0], 
                  # vmin=0, vmax=1, 
                  cmap='viridis', #'Greys',  
                  # interpolation='nearest'
                  )

fig.colorbar(mapp, ax=ax)
plt.show()
plt.close()


sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(num=3, figsize=(13,10))
ax = plt.subplot(111)

mapp = plt.imshow(track_w[130], 
                  # vmin=0, vmax=1, 
                  cmap='viridis', #'Greys',  
                  # interpolation='nearest'
                  )

fig.colorbar(mapp, ax=ax)

plt.show()
plt.close()



#%%
# plt.imshow(track_w[1], 
#             # vmin=0, vmax=1, 
#             # cmap='viridis', #'Greys',  
#             # interpolation='nearest'
#             )
# plt.show()
# plt.close()

# plt.imshow(track_w[2],
#             # vmin=0, vmax=1, 
#             # cmap='viridis', #'Greys',  
#             # interpolation='nearest'
#             )
# plt.show()
# plt.close()
