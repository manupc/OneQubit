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
from sklearn.datasets import make_blobs
import pickle

# Hyper-info
problem_name= 'Blobs6'


# Load data
X, y= make_blobs(random_state= 11, n_samples= 180, centers=6, center_box=(-10, 10), cluster_std=0.5)
print('clusters: {}'.format(len(np.unique(y))))

import matplotlib.pyplot as plt

colours= ['red', 'blue', 'black', 'orange', 'cyan', 'green', 'pink', 'gray']
for k, y_v in enumerate(np.unique(y)):
    idx= np.where(y==y_v)[0]
    x_v= X[idx]
    plt.scatter(x_v[:, 0], x_v[:, 1], color=colours[k])


x= X

# Escala de datos a [0, pi]
for column in range(x.shape[1]):
    mini, maxi= np.min(x[:, column]), np.max(x[:, column])
    x[:, column]= np.pi*(x[:, column] - mini)/(maxi - mini)





# Division in train and test
trPerc= 0.8

x_tr= []
x_ts= []


for y_i in np.unique(y):
    idx= np.where(y==y_i)[0]
    p= np.random.permutation(len(idx))
    idx= idx[p]
    n_tr= int(trPerc*len(idx))
    x_tr_i= x[ idx[:n_tr] ]
    x_ts_i= x[ idx[n_tr:] ]
    
    x_tr.append(x_tr_i)
    x_ts.append(x_ts_i)
    
x_tr= np.vstack(x_tr)
x_ts= np.vstack(x_ts)



data= {
       'train' : x_tr,
       'test' : x_ts
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


