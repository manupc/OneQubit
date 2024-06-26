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
from Evaluator import ClusteringEvaluator
from DataLoaders import load_Blobs6
from Model import model_builder


# Hyper-info
RESTART= False
DEBUG= False
opt_alg_name= 'ES'
problem_name= 'Blobs6'
model_type= 'QML'




def clustering_method_6_clusters(e_v : np.ndarray):

    y_pred= []
    
    for v in e_v:
        idx= np.argmax(np.abs(v))
        if v[idx]<0:
            idx+= 3
        y_pred.append(idx)
    return np.array(y_pred, dtype=int)



# Load data
data = load_Blobs6(data_folder)
x_tr= data['train']
x_ts= data['test']


nIn= x_tr.shape[1]
n_model_parameters= 6 # Number of model parameters



# Model parameters
model_param= {
    'solution': [], # Solution parameters
    'n_inputs': nIn, # Number of model inputs
    'input_rotations' : 'YZ', # Data encoding mechanism
    'n_layers': 1, # Number of model layers
    'prepare_state': True, # True to include a state preparation layer
    'observables': 'ZXY', # String containing the observables to be used
    'n_model_parameters' : n_model_parameters # Number of free parameters to be optimized
    }



# Creates the problem evaluator
evaluator_params= {
        'X': x_tr,  # Input data
        'model_builder': model_builder, # Callable to build a model from parameters
        'model_param' : model_param,
        'target_metric' : 1.0, # Target metric value
        'clustering_method': clustering_method_6_clusters
    }
evaluator= ClusteringEvaluator(evaluator_params)




# Callable to build a One-qubit classification model with 3 parameters for Iris Setosa
n_model_parameters= 6 # Number of model parameters
nInputs= 2 # Circuit Inputs
observables= 'ZXY' # Solution observables




# ES algorithm: Parameters
params= {}
params['mu']= 10
params['lmbda']= 50
params['lr']= 0.3 # Learning Rate
params['min_bound_value']= None
params['max_bound_value']= None
params['es_plus']= True # Use ES(mu+lambda) True; False for ES(mu, lambda)
params['initialize_method']= 'U' # 'U' for Univform, 'N' for normal
params['maximization_problem']= True
params['sol_size']= n_model_parameters
params['sol_evaluator']= evaluator


execution= {}
execution['MaxIterations']= 30
execution['MaxEvaluations']= None
execution['verbose']= True


store_results= ['iterations', 'evaluations', 'best', 'best_fitness',
                'time', 'history_mean_fitness', 'history_best_fitness', 
                'test_performance', 'test_predictions', 'train_predictions']

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


def calculate_test(results):
    
    test_evaluator_params= {
            'X': x_ts,  # Input data
            'model_builder': model_builder, # Callable to build a model from parameters
            'model_param' : model_param,
            'target_metric' : 1.0, # Target metric value
            'clustering_method': clustering_method_6_clusters
        }
    test_evaluator= ClusteringEvaluator(test_evaluator_params)
    best= results['best']
    test_performance, _= test_evaluator(best)
    test_predictions= test_evaluator.predict(best)
    train_predictions= evaluator.predict(best)
    
    results['test_performance']= test_performance
    results['test_predictions']= test_predictions
    results['train_predictions']= train_predictions
    print('Test results: {}'.format(test_performance))
    return results





runner= Experimenter(exprunner_param)
runner.run(force_restart=RESTART, post_processing_outputs=calculate_test, debug=DEBUG)
