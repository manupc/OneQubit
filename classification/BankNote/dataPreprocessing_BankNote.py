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
problem_name= 'BankNote'


# Load data
data = np.loadtxt(data_folder + "data_banknote_authentication.txt",delimiter=",", dtype=np.float32)
features= ['variance', 'skewness', 'curtosis', 'entropy']
x_raw= data[:, :-1]
y= data[:, -1].astype(int)
colours= ['red', 'blue']
markers= ['o', 's']
target_class= ['No', 'Yes']


# Escala de datos a [0, pi]
for column in range(x_raw.shape[1]):
    mini, maxi= np.min(x_raw[:, column]), np.max(x_raw[:, column])
    x_raw[:, column]= np.pi*(x_raw[:, column] - mini)/(maxi - mini)


"""
x= x_raw[:, :2].copy()
x[:, 0]= 0.5*x_raw[:, 0] + 0.5*x_raw[:, 1]
x[:, 1]= x_raw[:, 2]
"""
x= x_raw[:, :2].copy()
x[:, 0]= np.sqrt(x_raw[:, 0] + x_raw[:, 1])
x[:, 1]= np.sqrt(x_raw[:, 2])




# Division in train and test
trPerc= 0.8

x_tr, y_tr= [], []
x_ts, y_ts= [], []

for y_i in np.unique(y):
    idx= np.where(y==y_i)[0]
    p= np.random.permutation(len(idx))
    idx= idx[p]
    n_tr= int(trPerc*len(idx))
    x_tr_i= x[ idx[:n_tr] ]
    x_ts_i= x[ idx[n_tr:] ]
    y_tr_i= y[ idx[:n_tr] ]
    y_ts_i= y[ idx[n_tr:] ]
    
    x_tr.append(x_tr_i)
    x_ts.append(x_ts_i)
    y_tr.append(y_tr_i)
    y_ts.append(y_ts_i)
    
x_tr= np.vstack(x_tr)
x_ts= np.vstack(x_ts)
y_tr= np.concatenate(y_tr)
y_ts= np.concatenate(y_ts)


data= {
       'train' : (x_tr, y_tr),
       'test' : (x_ts, y_ts)
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


