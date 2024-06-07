#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:19:55 2024

@author: manupc
"""

import sys
import pathlib
current_folder= str(pathlib.Path(__file__).parent.resolve())+'/'
sys.path.insert(1, current_folder+'../../common/')
data_folder= current_folder+'../../data/'






import random
import numpy as np
from DQN import DQN as RL_alg
from Experimenter import Experimenter
import tensorflow as tf
from DataLoaders import load_FrozenLakeOneHot


RESTART= False
DEBUG= False

# Experimentation
algorithmName= 'DQN'
dataName= 'FrozenLake'
method= ''


env_data= load_FrozenLakeOneHot()
env_n_tests= env_data['n_tests']
env_builder= env_data['env_builder']
reward_solved= env_data['solved']



"""
Creates a FF network with nInputs inputs, nOutputs outputs, and a list of layers Hidden_structure=[(n,act)] 
with n=the number of neurons in the hidden layer with act=activation function
"""
def CreateClassicModelStructure(nInputs, nOutputs, hidden_structure):
    
    # Weight Initialized
    model= tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(nInputs)))
    for layer in hidden_structure:
        #model.add(tf.keras.layers.Dense(layer[0], activation=layer[1], kernel_initializer=initializer),)
        model.add(tf.keras.layers.Dense(layer[0], activation=layer[1]))
    model.add(tf.keras.layers.Dense(nOutputs))
    return model

# Create model
aux_env= env_builder()
nInputs= aux_env.nInputs
nOutputs= aux_env.action_space.n

print('inputs: {}, outputs= {}'.format(nInputs, nOutputs))

model_structure= [[100, 'relu'], [100, 'relu']] # Hidden layers of feedforward model
optimizer= tf.keras.optimizers.Adam(learning_rate= 0.005)
model_builder= lambda : CreateClassicModelStructure(nInputs, nOutputs, model_structure)


params= {}

MaxIterations= None # Maximum number of DQN iterations to run
MaxEpisodes= 50000 # Maximum number of episodes to run

# Main DQN parameters
params['env_builder']= env_builder #: Callable to build the environment
params['gamma']= 0.99 #: Discount factor
params['model_builder']= model_builder #: callable to create a policy that returns logits
params['optimizer']= optimizer #: Optimization algorithm
params['batch_size']= 128 #: Batch Size
params['training_envs']= 2 #: Number of environments to populate buffer (default= 1)
params['DoubleDQN']= True #: True to activate DoubleDQN rule, False to set usual DQN
params['mean_history_horizon']= 100 #: Number of past episodes to consider in the history of mean training returns

# Replay Buffer parameters
params['buffer']= {}
params['buffer']['type']= 'deque' #: Type of buffer ('deque')
params['buffer']['capacity']= 50000 #: Size of the buffer
params['buffer']['populate']= False #: True to populate the buffer initially, False otherwise

# Exploration policy
params['exploration']= {}
params['exploration']['type']= 'eGreedy' #: Type of exploration (default='eGreedy')
params['exploration']['eps0']= 0.8 #: Initial epsilon for e-Greedy policy 
params['exploration']['epsf']= 0.05 #: Final epsilon for e-Greedy policy 
params['exploration']['decrease_type']= 'linear' #: 'linear' for linear eps decrease
params['exploration']['update_type']= 'episodes' #: e-Greedy update with 'iterations' or 'episodes'
params['exploration']['eps_steps']= int(0.8*MaxEpisodes) #: e-Greedy steps to reach eg_epsf from eg_eps0

# Target network update method
params['target_update']= {}
params['target_update']['type']= 'soft' #: 'hard' for hard update/'soft' for soft update
params['target_update']['alpha']= 0.1 



# Test settings
params['test']= {}
params['test']['reference']= 'episodes' #: None, or int to set the number of algorithm iterations to wait before testing
params['test']['steps']= 5 #: None, or int to set the number of algorithm iterations to wait before testing
params['test']['size']= env_n_tests #: Number of test episodes 
params['test']['reward_solved']= reward_solved #: Reward value to consider environment solved, or None




execution= {}
execution['MaxIterations']= MaxIterations
execution['MaxEpisodes']= MaxEpisodes
execution['verbose']= True


store_results= ['iterations', 'episodes', 'best', 'best_fitness',
                'time', 'history_mean_fitness', 'history_best_fitness', 
                'history_loss']
seed_initializer= lambda x: (np.random.seed(x), random.seed(x))

exprunner_param= {}
exprunner_param['algorithm']= RL_alg                   # Algorithm's Python class with 'run' method
exprunner_param['alg_params']= params              # Dictionary containing algorithm construction method
exprunner_param['run_params']= execution           # Dictionary containing parameters for the run method
exprunner_param['algorithm_name']= algorithmName        #: Name of the algorithm
exprunner_param['problem_name']= dataName        #: Name of the problem to be solved
exprunner_param['additional_info']= method             #: Text containing additional info for the name of the output file
exprunner_param['runs']= 30                 #: Number of runs
exprunner_param['store_results']= store_results    #: List containing names of results to be saved
exprunner_param['output_file']= None               #: Output file for results (no extension). Default None
exprunner_param['seed_initializer']= seed_initializer #: Callable with input parameter the seed


runner= Experimenter(exprunner_param)

def serialize_model(results):
    best= results['best']
    serialized= []
    for var in best.trainable_variables:
        serialized.append(var.numpy())
    results['best']= serialized
    return results


runner.run(force_restart=False, post_processing_outputs=serialize_model, debug=DEBUG)






