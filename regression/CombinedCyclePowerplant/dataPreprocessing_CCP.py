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
import pandas as pd

# Hyper-info
problem_name= 'CCP'


# Load data
data = np.loadtxt(data_folder + "CCP.csv", delimiter=",", dtype=str)

data= data[1:, :].astype(np.float64)




X_raw= np.empty(data.shape)
for i in range(data.shape[1]):
    mini, maxi= np.min(data[:, i]), np.max(data[:, i])
    X_raw[:, i]= (data[:, i] - mini)/(maxi-mini)
X_raw[:, :-1]= X_raw[:, :-1]*np.pi
X_raw[:, -1]= 2*X_raw[:, -1]-1

x= X_raw[:, :2]
x[:, 0]=  X_raw[:, 0] +  X_raw[:, 1]
x[:, 1]=  X_raw[:, 2] +  X_raw[:, 3]
for i in range(x.shape[1]):
    mini, maxi= np.min(x[:, i]), np.max(x[:, i])
    x[:, i]= (x[:, i] - mini)/(maxi-mini)

y= X_raw[:, -1]




# Division in train and test
trPerc= 0.8


n_tr= int(trPerc*len(y))
x_tr= x[:n_tr, :]
x_ts= x[n_tr:, :]
y_tr= y[:n_tr]
y_ts= y[n_tr:]


data= {
       'train' : (x_tr, y_tr),
       'test' : (x_ts, y_ts)
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


