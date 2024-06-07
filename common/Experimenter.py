#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:05:07 2024

@author: manupc
"""

import os
import numpy as np
import pickle

"""
Class to perform automatic experimentation and result logging
"""
class Experimenter:
    
    """
    params: Dictorionary containing:
        params['algorithm']: Algorithm's Python class with 'run' method
        params['alg_params']: Dictionary containing algorithm construction method
        params['run_params']: Dictionary containing parameters for the run method
        params['algorithm_name']: Name of the algorithm
        params['problem_name']: Name of the problem to be solved
        params['additional_info']: Text containing additional info for the name of the output file
        params['runs']: Number of runs
        params['store_results']: List containing names of results to be saved
        params['output_file']: Output file for results (no extension). Default None
        params['seed_initializer']: Callable with input parameter the seed
    """
    def __init__(self, params):
        self.__alg= params['algorithm']
        self.__alg_params= params['alg_params']
        self.__run_params= params['run_params']
        self.__alg_name= params['algorithm_name']
        self.__problem_name= params['problem_name']
        self.__info= params['additional_info']
        self.__runs= params['runs']
        self.__output_file= params['output_file']
        self.__seed_initializer= params['seed_initializer']
        self.__store_results= params['store_results']
        
        if self.__output_file is None:
            self.__output_file= self.__alg_name+'_'+self.__problem_name
            if len(self.__info)>0:
                self.__output_file+= '_'+self.__info
        if not self.__output_file.endswith('.pkl'):
            self.__output_file+= '.pkl'
            
            
            
    def __output_file_exists(self):
        file_exists= os.path.isfile(self.__output_file)
        return file_exists


    def __load_output_file(self):
        with open(self.__output_file, 'rb') as f:
            file_content= pickle.load(f)
        return file_content['seeds'], file_content['done'], file_content['results']

    
    def __save_output_file(self, seeds, done, experiment, results):
        
        data= {}
        if self.__output_file_exists():
            seeds, done, exper= self.__load_output_file()
            
            data['seeds']= seeds
            data['done']= done
            data['results']= exper
        else:
            data['seeds']= seeds
            data['done']= done
            data['results']= {}
            
        if experiment is not None:
            data['results'][experiment]= results
            if len(results) > 0:
                data['done'][experiment]= True
            else:
                data['done'][experiment]= False
        else:
            data['results']= {}

        with open(self.__output_file, 'wb') as f:
            pickle.dump(data, f)


    def run(self, force_restart= False, post_processing_outputs= None, debug= False):
        
        if self.__output_file_exists() and not force_restart:
            seeds, done, _= self.__load_output_file()
        else:
            self.__exper= {}
            seeds= np.random.randint(low= 0, high= 100000, size=self.__runs).tolist()
            done= [False]*self.__runs
            self.__save_output_file(seeds, done, None, None)
            
        
        verbose_add= self.__alg_name+' '+self.__problem_name+' '+self.__info
        alg= self.__alg
        params= self.__alg_params
        run_params= self.__run_params
        experiment= 0
        while experiment < self.__runs:

            if done[experiment]:
                print('Experiment {} already exists for {}'.format(experiment+1, verbose_add))
                experiment+= 1
                continue
        
            current_seed= seeds[experiment]
            print('\n\n#############################################')
            print('Running experiment {}/{} for {}. Seed= {}'.format(experiment+1, self.__runs, verbose_add, current_seed))
            print('#############################################\n\n')
            
            # Set random seed
            self.__seed_initializer(current_seed)

            # Instantiate algorithm
            alg_instance= alg(params)
            if debug:
                run_params['verbose_text_append']= 'DEBUG MODE '+verbose_add+' Exper {}/{}'.format(experiment+1, self.__runs)
                out= alg_instance.run(run_params)
                print('Debug execution finished successfully')
                raise Exception('Debug finished')
                
            # Run algorithm
            try:
                run_params['verbose_text_append']= verbose_add+' Exper {}/{}'.format(experiment+1, self.__runs)
                out= alg_instance.run(run_params)
                
                if post_processing_outputs is not None:
                    out= post_processing_outputs(out)

                error= False
            except KeyboardInterrupt:
                print('Experiment stopped')
                return
            except:
                print('Something went wrong in Experiment {}'.format(experiment))
                error= True
            
            if not error:
                # Gather results
                result= {}
                for key in self.__store_results:
                    result[key]= out[key]
            
                self.__save_output_file(seeds, done, experiment, result)
                experiment+= 1

            
    def getResults(self):
        if self.__output_file_exists():
            _, _, results= self.__load_output_file()
        else:
            results= None
        return results
            
        