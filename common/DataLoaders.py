#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:37:35 2024

@author: manupc
"""

import pickle
import numpy as np 
import gymnasium as gym


# Load the Iris Setosa dataset
def load_IrisSetosa(folder : str):
    with open(folder + 'IrisSetosa.pkl', 'rb') as f:
        d= pickle.load(f)
    return d


# Load the Iris dataset
def load_Iris(folder : str):
    with open(folder + 'Iris.pkl', 'rb') as f:
        d= pickle.load(f)
    return d


# Load the BalanceScale dataset
def load_BalanceScale(folder : str):
    with open(folder + 'BalanceScale.pkl', 'rb') as f:
        d= pickle.load(f)
    return d


# Load the BankNote dataset
def load_BankNote(folder : str):
    with open(folder + 'BankNote.pkl', 'rb') as f:
        d= pickle.load(f)
    return d



# Load the AirPassengers dataset
def load_AirPassengers(folder : str, time_horizon : int):
    with open(folder + 'AirPassengersT{}.pkl'.format(time_horizon), 'rb') as f:
        d= pickle.load(f)
    return d


# Load the CCP dataset
def load_CCP(folder : str):
    with open(folder + 'CCP.pkl', 'rb') as f:
        d= pickle.load(f)
    return d


# Load the OikoLab dataset
def load_OikoLab(folder : str, time_horizon : int):
    with open(folder + 'OikoLabT{}.pkl'.format(time_horizon), 'rb') as f:
        d= pickle.load(f)
    return d


# Load the OldFaithful dataset
def load_OldFaithful(folder : str):
    with open(folder + 'OldFaithful.pkl', 'rb') as f:
        d= pickle.load(f)
    return d


# Load the Blobs dataset
def load_Blobs(folder : str):
    with open(folder + 'Blobs.pkl', 'rb') as f:
        d= pickle.load(f)
    return d

# Load the Blobs dataset with 6 clusters
def load_Blobs6(folder : str):
    with open(folder + 'Blobs6.pkl', 'rb') as f:
        d= pickle.load(f)
    return d




class CartPoleObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.nInputs= 2
        self.nOutputs= env.action_space.n

   
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        new_obs= np.empty(2, dtype=np.float32)
        new_obs[0]= np.pi*((obs[2]/0.418)+1)/2
        new_obs[1]= np.pi*(2*np.arctan(obs[3])/np.pi + 1)/2
        return new_obs

env_builder_cartpole= lambda render=False: CartPoleObervationWrapper(gym.make('CartPole-v1', render_mode='rgb_array' if render else None))



# Load the CartPole env
def load_CartPole():
    d= {
        'env_builder' : env_builder_cartpole,
        'n_tests' : 100,
        'solved' : 500,
        'actions': 2
        }
    return d




"""
Observation Wrapper for FrozenLake env
"""
class FrozenLakeObervationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space= gym.spaces.Box(low=0, high=np.pi, shape=(2,), dtype=np.float32)
        self.nInputs= 2
        self.nOutputs= env.action_space.n
        self.rvalues= np.linspace(start= 0, stop= np.pi, num=4)
    
    
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        
        row= obs//4
        col= obs % 4
        new_obs= np.zeros(2, dtype= np.float32)
        new_obs[0]= self.rvalues[col]
        new_obs[1]= self.rvalues[row]
        return new_obs

env_builder_fl= lambda render=False: FrozenLakeObervationWrapper(gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array' if render else None))

env_builder_fl_slippery= lambda render=False: FrozenLakeObervationWrapper(gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array' if render else None))



# Load the FrozenLake env
def load_FrozenLake():
    d= {
        'env_builder' : env_builder_fl,
        'n_tests' : 1,
        'solved' : 1,
        'actions': 4
        }
    return d



# Load the FrozenLake env
def load_FrozenLakeSlippery():
    d= {
        'env_builder' : env_builder_fl_slippery,
        'n_tests' : 100,
        'solved' : 0.74, #0.54,
        'actions': 4
        }
    return d





"""
Observation Wrapper for FrozenLake env One-Hot encoded
"""
class FrozenLakeOneHotObervationWrapper(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space= gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.nInputs= 4
        self.nOutputs= env.action_space.n
    
    
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):
        
        b_r= bin(obs)[2:].rjust(4, '0')
        new_obs= np.zeros(4, dtype= np.float32)
        for i in range(len(b_r)):
            if b_r[i] == '1':
                new_obs[i]= 1.0
        return new_obs

env_builder_fl_oh= lambda render=False: FrozenLakeOneHotObervationWrapper(gym.make('FrozenLake-v1', is_slippery=False, render_mode='rgb_array' if render else None))

env_builder_fl_slippery_oh= lambda render=False: FrozenLakeOneHotObervationWrapper(gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array' if render else None))



# Load the FrozenLake env
def load_FrozenLakeOneHot():
    d= {
        'env_builder' : env_builder_fl_oh,
        'n_tests' : 1,
        'solved' : 1,
        'actions': 4
        }
    return d


# Load the FrozenLake env
def load_FrozenLakeSlipperyOneHot():
    d= {
        'env_builder' : env_builder_fl_slippery_oh,
        'n_tests' : 100,
        'solved' : 0.74, #1,
        'actions': 4
        }
    return d




"""
Observation Wrapper for MountainCar env
"""
class MountainCarObervationWrapper(gym.ObservationWrapper):
    
    """
    env: Entorno a encapsular
    """
    def __init__(self, env):
        super().__init__(env)
        self.nInputs= env.observation_space.shape[0]
        self.nOutputs= env.action_space.n
        self.observation_space= gym.spaces.Box(low=0, high=np.pi, shape=(2,), dtype=np.float32)

   
    """
    Transformación de la observación obs en One Hot
    """
    def observation(self, obs):

        new_obs= np.empty(self.nInputs, dtype=np.float32)
        new_obs[0]= np.pi*(obs[0]+1.2)/1.8 - np.pi/2
        new_obs[1]= np.pi*(obs[1]+0.07)/0.14 - np.pi/2
        return new_obs


env_builder_mc= lambda render=False: MountainCarObervationWrapper(gym.make('MountainCar-v0', render_mode='rgb_array' if render else None))


# Load the MountainCar env
def load_MountainCar():
    d= {
        'env_builder' : env_builder_mc,
        'n_tests' : 100,
        'solved' : -110,
        'actions' : 3
        }
    return d
