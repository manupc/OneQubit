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
problem_name= 'OikoLabT4'


# Load data
df= pd.read_csv(data_folder + 'oikolab.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
df = df.set_index(['DateTime'])
df= df.resample("1M").mean()
data= df['Temperature (ºC)'].to_numpy().squeeze()


data= (data-np.min(data))/(np.max(data) - np.min(data))


n_inputs= 4

X_raw= []
for i in range(len(data)-n_inputs):
    pattern= data[i:(i+n_inputs+1)]
    X_raw.append(pattern)
X_raw= np.array(X_raw)
Y= X_raw[:, -1]
X_raw= X_raw[:, :-1]

Y= 2*Y-1
X_raw= X_raw*np.pi

x= X_raw.reshape(-1, n_inputs)
y= Y.reshape(-1)




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


