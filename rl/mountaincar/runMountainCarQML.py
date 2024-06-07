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




import random
from Experimenter import Experimenter
from ES_Mu_Lambda import ES as opt_algorithm
import numpy as np
from Evaluator import RLEvaluator
from DataLoaders import load_MountainCar
from Model import model_builder

# Hyper-info

## BUENA SEMILLA: 50550, 29302, 25174
RESTART= False
DEBUG= False
opt_alg_name= 'ES'
problem_name= 'MountainCar'
model_type= 'QML'


def ActionSelector(logits):
    if not isinstance(logits, np.ndarray):
        logits= np.array(logits)
    logits= logits.reshape(-1, 3)
    actions= np.argmax(np.abs(logits), axis=1)
    return actions





env_data= load_MountainCar()
env_n_tests= env_data['n_tests']
env_builder= env_data['env_builder']
reward_solved= env_data['solved']



# Model parameters
model_param= {
    'solution': [], # Solution parameters
    'n_inputs': 2, # Number of model inputs
    'input_rotations' : 'YZ', # Data encoding mechanism
    'n_layers': 1, # Number of model layers
    'prepare_state': True, # True to include a state preparation layer
    'observables': 'XYZ', # String containing the observables to be used
    'n_model_parameters' : 6 # Number of free parameters to be optimized
    }





# Creates the problem evaluator
evaluator_params= {
        'env_builder': env_builder,  # Callable to build an environment containing fields nInputs and nOutputs
        'test_seeds' : False, # True to set constant test seeds, False to leave test seeds free
        'nTests' : env_n_tests, # Number of tests to assess a solution performance
        'action_selector' : ActionSelector, # callable to select an action from model output
        'reward_solved' : reward_solved,
        'model_builder': model_builder, # Callable to build a model from parameters
        'model_param' : model_param
    }
evaluator= RLEvaluator(evaluator_params)



# ES algorithm: Parameters
params= {}
params['mu']= 3
params['lmbda']= 15
params['lr']= 0.1 # Learning Rate
params['min_bound_value']= None
params['max_bound_value']= None
params['es_plus']= True # Use ES(mu+lambda) True; False for ES(mu, lambda)
params['initialize_method']= 'U' # 'U' for Univform, 'N' for normal
params['maximization_problem']= True
params['sol_size']= model_param['n_model_parameters']
params['sol_evaluator']= evaluator


execution= {}
execution['MaxIterations']= 50
execution['MaxEvaluations']= None
execution['verbose']= True


store_results= ['iterations', 'evaluations', 'best', 'best_fitness',
                'time', 'history_mean_fitness', 'history_best_fitness']

seed_initializer= lambda x: (np.random.seed(x), random.seed(x))
exprunner_param= {}
exprunner_param['algorithm']= opt_algorithm                   # Algorithm's Python class with 'run' method
exprunner_param['alg_params']= params              # Dictionary containing algorithm construction method
exprunner_param['run_params']= execution           # Dictionary containing parameters for the run method
exprunner_param['algorithm_name']= opt_alg_name        #: Name of the algorithm
exprunner_param['problem_name']= problem_name        #: Name of the problem to be solved
exprunner_param['additional_info']= model_type # Model type that solves the problem
exprunner_param['runs']= 30                 #: Number of runs
exprunner_param['store_results']= store_results    #: List containing names of results to be saved
exprunner_param['output_file']= None               #: Output file for results (no extension). Default None
exprunner_param['seed_initializer']= seed_initializer #: Callable with input parameter the seed


runner= Experimenter(exprunner_param)

runner.run(force_restart=RESTART, debug=DEBUG)

