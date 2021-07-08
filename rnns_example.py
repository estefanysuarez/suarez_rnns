# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:13:24 2021

@author: Estefany Suarez
"""

import torch
from torch import nn

import numpy as np

from rnns import rnns

#%%
text = ['hey how are you','good i am fine','have a nice day']

# Join all the sentences together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))

# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}


#%%
maxlen = len(max(text, key=len))
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += ' '
        

#%%
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])
    
    # Remove firsts character for target sequence
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
    
    
#%%
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]


#%%
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

#%%
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#%%
# Model parameters
input_size = dict_size
output_size = dict_size
hidden_size = 12

#%%
# Connectivity matrix
# win = np.random.randint(0, 2, (hidden_size, input_size)).astype(float)
win = np.random.rand(hidden_size, input_size)
w = np.random.rand(hidden_size, hidden_size)*np.random.randint(0,2,(hidden_size, hidden_size))

#%%
# Instantiate model
model = rnns.NeuralNet(input_size=input_size, 
                       output_size=output_size, 
                       hidden_size=hidden_size,  
                       # win=win,
                       # w=w,
                        device=device
                       )

model.to(device)

#%%
# Model hyper-parameters
n_epochs = 100
lr = 0.01

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%%
# Model training 
for epoch in range(1, n_epochs + 1):
    
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    # input_seq = input_seq.to(device)
    
    output, hidden = model(input_seq.float())
    
    output = output.to(device)
    target_seq = target_seq.to(device)
    
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    
    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
