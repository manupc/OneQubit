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
from sklearn.datasets import load_iris
import pickle

# Hyper-info
problem_name= 'Iris'


# Load Iris data
data= load_iris()
setosa, versicolor, virginica= 0, 1, 2
attr_ampl, attr_phase= 2, 3 # Seleccion de largo y ancho de petalo
x_raw, y_raw= data['data'], data['target']



# Escala de datos a [0, pi]
x= x_raw[:, [attr_ampl, attr_phase]]
y= y_raw
for column in range(x.shape[1]):
    mini, maxi= np.min(x[:, column]), np.max(x[:, column])
    x[:, column]= np.pi*(x[:, column] - mini)/(maxi - mini)


# Division en train and test
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
    
x_tr= np.vstack(x_tr).reshape(-1, 2)
x_ts= np.vstack(x_ts).reshape(-1, 2)
y_tr= np.concatenate(y_tr).reshape(-1)
y_ts= np.concatenate(y_ts).reshape(-1)




data= {
       'train' : (x_tr, y_tr),
       'test' : (x_ts, y_ts)
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


