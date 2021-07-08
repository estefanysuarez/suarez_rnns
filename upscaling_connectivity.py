# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:48:24 2021

@author: Estefany Suarez
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rnns import connectivity

#%%
# load connectivity matrix
w = np.load('F:/Data/Lausanne/Lausanne_npy_ind/struct/struct_num_scale033.npy').mean(axis=2).astype(int) #load connectivity matrix
mask = np.loadtxt('F:/Data/Lausanne/Lausanne_txt_avg/struct_num/bin/weights033.txt').astype(int) #load binary mask
np.fill_diagonal(mask, 1)
w = w*mask # apply mask to select only consistent structural connections


#%%
# upscale connectivity matrix
density = 2*np.ones(len(w), dtype=int) #np.random.randint(1,10,len(conn)) #
neuron_ids_per_roi, scld_w, scld_mask = connectivity.upscale_rnn(original_network=w,
                                                                 fraction_intrinsic=None,
                                                                 neuron_density=density,
                                                                 intrinsic_sparsity=0.8,
                                                                 extrinsic_sparsity=0.2,
                                                                 allow_self_conns=True,
                                                                 return_mask=True,
                                                                 )


#%%
test = np.zeros_like(scld_w)
test[np.triu_indices_from(scld_w, 1)] = scld_w[np.triu_indices_from(scld_w, 1)]
test = test.T
test[np.triu_indices_from(scld_w, 1)] = scld_w[np.triu_indices_from(scld_w, 1)]

#%%
sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(num=1, figsize=(13,10))
ax = plt.subplot(111)

mapp = plt.imshow(w,
                  vmin=0,
                  vmax=2500,
                  cmap='viridis', #'Greys',
                  # interpolation='nearest'
                  )

fig.colorbar(mapp, ax=ax)
plt.show()
plt.close()

fig.savefig('C:/Users/User/Dropbox/figs/w', transparent=True, bbox_inches='tight', dpi=300)


sns.set(style="ticks", font_scale=2.0)
fig = plt.figure(num=2, figsize=(13,10))
ax = plt.subplot(111)

mapp = plt.imshow(test,
                  # vmin=0,
                   vmax=40000,
                  cmap='viridis', #'Greys',
                  # interpolation='nearest'
                  )

fig.colorbar(mapp, ax=ax)
plt.show()
plt.close()

fig.savefig('C:/Users/User/Dropbox/figs/scld_w', transparent=True, bbox_inches='tight', dpi=300)



#%%
# # binary connectivity matrix
# # plt.imshow(mask.astype(int), vmin=0, vmax=1, cmap='Greys',  interpolation='nearest')
# plt.spy(mask.astype(int))
# plt.show()
# plt.close()

# # plt.imshow(scld_mask.astype(int), vmin=0, vmax=1, cmap='Greys',  interpolation='nearest')
# plt.spy(scld_mask.astype(int))
# plt.show()
# plt.close()


#%%
# binary connectivity matrix

# plt.imshow(scaled.astype(bool).astype(int), vmin=0, vmax=1, cmap='Greys',  interpolation='nearest')
# plt.show()
# plt.close()
