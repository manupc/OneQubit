#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:37:35 2024

@author: manupc
"""

import sys
import pathlib
current_folder= str(pathlib.Path(__file__).parent.resolve())+'/'
sys.path.insert(1, current_folder+'../../common/')
data_folder= current_folder+'../../data/'



import numpy as np
import pickle

# Hyper-info
problem_name= 'OldFaithful'


# Load data
data = np.loadtxt(data_folder + "faithful.csv", delimiter=",", dtype=str)
data= data[1:, 1:].astype(np.float32)

x_raw= data

# Shuffle data
x= x_raw[ np.random.permutation(len(x_raw)) ]

# Data scaling to [0, pi]
for column in range(x.shape[1]):
    mini, maxi= np.min(x[:, column]), np.max(x[:, column])
    x[:, column]= np.pi*(x[:, column] - mini)/(maxi - mini)





# Division in train and test
trPerc= 0.8

n_tr= int(trPerc*len(x))

x_tr= x[:n_tr]
x_ts= x[n_tr:]


data= {
       'train' : x_tr,
       'test' : x_ts
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


