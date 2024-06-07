#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:10:40 2024

@author: manupc
"""

from OneQubitVQC import OneQubitVQC


"""
    Creates the one-qubit model
    INPUTS:
        info: Dictionary containing keys:
            'solution': Solution parameters
            'nInputs': Number of model inputs
            'prepare_state': True to include a state preparation layer
            'input_rotations': Input rotations per layer
            'observables': String containing the observables to be used
    OUTPUTS: A one qubit VQC representing the model
"""
def model_builder(info : dict):
    
    parameters= info['solution']
    n_layers= info['n_layers']
    n_inputs= info['n_inputs']
    input_rots= info['input_rotations']
    state_preparation= info['prepare_state']
    observables= info['observables']
    
    
    if state_preparation: 
        structure= [
             # Layer 1
             [ '', # inputs
               [('Y', parameters[0]), ('Z', parameters[1]), ('X', parameters[2])] # Rotations
             ]]
        k= 3
    else:
        structure= []
        k= 0
    
    for i in range(n_layers):
        layer= [ input_rots, # inputs
          [('Y', parameters[k]), ('Z', parameters[k+1]), ('X', parameters[k+1])]
        ]
        k+= 3
        structure.append(layer)

    model= OneQubitVQC(nInputs= n_inputs, structure= structure, output_observables= observables)
    return model
