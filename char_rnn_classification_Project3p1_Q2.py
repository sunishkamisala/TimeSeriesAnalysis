#!/usr/bin/env python
# coding: utf-8

# ## DSC 275/475: Time Series Analysis and Forecasting (Fall 2020) 
# 
#  ## Project 3.2–Sequence Classification with Recurrent Neural Networks 

# Sample Code Block for Problem 1.1

# In[4]:

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import pandas as pd
import unicodedata
import string
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# In[5]:


def findFiles(path): 
    return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# In[6]:


names = {}
languages = []


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# (TO DO:) CHANGE FILE PATH AS NECESSARY
for filename in findFiles('/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    languages.append(category)
    lines = readLines(filename)
    names[category] = lines


# In[7]:


n_categories = len(languages)

def letterToIndex(letter):
    return all_letters.find(letter)


def nameToTensor(name):
    tensor = torch.zeros(len(name), 1, n_letters)
    for li, letter in enumerate(name):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# In[54]:


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, N_LAYERS,OUTPUT_SIZE):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE, # number of hidden units
            num_layers = N_LAYERS, # number of layers
            batch_first = True)
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
            
    def forward(self, x):
        r_out, h = self.rnn(x, None) # None represents zero initial hidden state           
        out = self.out(r_out[:, -1, :])
        return out


# In[8]:


n_hidden = 128

allnames = [] # Create list of all names and corresponding output language
for language in list(names.keys()):
    for name in names[language]:
        allnames.append([name, language])
        
## (TO DO:) Determine Padding length (this is the length of the longest string) 

# maxlen = ..... # Add code here to compute the maximum length of string        
                
n_letters = len(all_letters)
n_categories = len(languages)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i.item()
    return languages[category_i], category_i

           


# In[13]:


learning_rate = 0.005
rnn = RNN(n_letters, 128, 1, n_categories)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)   # optimize all rnn parameters
loss_func = nn.CrossEntropyLoss()  

for epoch in range(5):  
    batch_size = len(allnames)
    random.shuffle(allnames)
    
    # if "b_in" and "b_out" are the variable names for input and output tensors, you need to create those
    
    b_in = ...  # (TO DO:) Initialize "b_in" to a tensor with size of input (batch size, padded_length, n_letters)
    b_out = ...  # (TO DO:) Initialize "b_out" to tensor with size (batch_size, n_categories, dtype=torch.long)       


    # (TO DO:) Populate "b_in" tensor 

    # (TO DO:) Populate "b_out" tensor
    
       

    output = rnn(b_in)                               # rnn output
    #(TO DO:)
    loss = loss_func(output, ....)   # (TO DO:) Fill "...." to calculate the cross entropy loss
    optimizer.zero_grad()                           # clear gradients for this training step
    loss.backward()                                 # backpropagation, compute gradients
    optimizer.step()                                # apply gradients
        
    # Print accuracy
    test_output = rnn(b_in)                   # 
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    test_y = torch.max(b_out, 1)[1].data.numpy().squeeze()
    accuracy = sum(pred_y == test_y)/batch_size
    print("Epoch: ", epoch, "| train loss: %.4f" % loss.item(), '| accuracy: %.2f' % accuracy)
    






