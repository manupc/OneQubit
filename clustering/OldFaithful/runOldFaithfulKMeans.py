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
import numpy as np
from DataLoaders import load_OldFaithful
from SKLModels import SKLKMeans as opt_algorithm



# Hyper-info
RESTART= True
DEBUG= False
opt_alg_name= 'KMeans'
problem_name= 'OldFaithful'
model_type= ''



# Load data
data = load_OldFaithful(data_folder)
x_tr= data['train']
x_ts= data['test']




# algorithm: Parameters
params= {}
params['Xtr']= x_tr #: Input training data
params['Xts']= x_ts #: Input test data
params['n_clusters']= 2
params['metric']= 'silhouette' #: 'perc_error' / 'perc_accuracy'
params['parameters']= None # Solution parameters if no need to train


execution= {}
execution['MaxIterations']= 1000
execution['verbose']= True


store_results= ['iterations', 'best', 'best_fitness', 'test_performance', 'time',
                'test_predictions', 'train_predictions']

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
