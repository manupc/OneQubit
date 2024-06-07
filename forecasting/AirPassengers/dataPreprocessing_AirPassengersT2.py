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
from scipy.optimize import curve_fit
import pickle

# Hyper-info
problem_name= 'AirPassengersT2'


# Load data
data = np.loadtxt(data_folder + "AirPassengers.csv", delimiter=",", dtype=str)

series= data[1:, 1].astype(np.float32)
series= np.log(series) # Remove variance variability

# Trend model
def linear_model(t, w0, w1):
    return t*w1+w0

t= np.array(list(range(len(series))))
w, _= curve_fit(linear_model, t, series)
print('Trend model parameters: ', w)

# Trend component
l_m= np.array([linear_model(x, w[0], w[1]) for x in t])


# Time Series Trend removal
t_series= series-l_m

# Data scale
data= (t_series - np.min(t_series))/ (np.max(t_series) - np.min(t_series))




n_inputs= 2

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
       'test' : (x_ts, y_ts),
       'trend_model' : w
       }

data_file= data_folder+problem_name+'.pkl'
with open(data_file, 'wb') as f:
    pickle.dump(data, f)


