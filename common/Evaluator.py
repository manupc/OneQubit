#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:36:54 2024

@author: manupc
"""

import numpy as np


class TSEvaluator:
    
        
    
    """
    INPUT: dictionary params containing:
        params['X']:  # Input data
        params['Y']:  # Output data
        params['model_builder']:  Callable to build a model from parameters
   """
    def __init__(self, params):
        self.__params= params
        self.reset()


    """
    Resets the evaluator to defaults
    """
    def reset(self):
        params= self.__params
        
        self.__X= params['X']
        self.__Y= params['Y']
        self.__model_builder= params['model_builder']
        self.__model_param= params['model_param']


    def problem_solved(self):
        return False


    def getModelForSolution(self, solution):
        model_param= self.__model_param
        model_param['solution']= solution
        return self.__model_builder(model_param)

    
    
    def predict(self, solution):
        model= self.getModelForSolution(solution)
        X= self.__X.copy()
        y_pred= model(X).numpy().reshape(-1)
        return y_pred
    
    
    
    def __call__(self, solution):
        
        y_pred= self.predict(solution)
        error= np.mean( (self.__Y - y_pred)**2 )
        return error, 1
    
    






class RLEvaluator:
    
        
    
    """
    INPUT: dictionary params containing:
        params['env_builder']: Callable to build an environment containing fields nInputs and nOutputs
        params['test_seeds']: True to set constant test seeds, False to leave test seeds free
        params['nTests']: Number of tests to assess a solution performance
        params['action_selector']': callable containing a batch of FQRS outputs to select an action
        params['reward_solved']: Avg Return to set if the problem is solved. None if not applicable
        params['model_builder']: Callable to build a model from parameters
        params['model_param']: Parameters to create the model
    """
    def __init__(self, params):
        self.__params= params
        self.reset()


    """
    Resets the evaluator to defaults
    """
    def reset(self):
        params= self.__params
        
        self.__env_builder= params['env_builder']
        self.__test_seeds= params['test_seeds'] if 'test_seeds' in params.keys() else True
        self.__nTests= params['nTests']  if 'nTests' in params.keys() else 100
        self.__action_selector= params['action_selector']
        self.__reward_solved= params['reward_solved'] if 'reward_solved' in params.keys() else None
        self.__model_builder= params['model_builder']
        self.__model_param= params['model_param']
        
        self.__problem_solved= False
        # Test envs
        self.__tsEnvs= [self.__env_builder() for _ in range(self.__nTests)]
        
        self.__testCounter= np.random.randint(low=1, high=100000)


    def problem_solved(self):
        if self.__reward_solved is None:
            return False
        return self.__problem_solved
        



    def getModelForSolution(self, solution):
        model_param= self.__model_param
        model_param['solution']= solution
        return self.__model_builder(model_param)
    

    def __call__(self, solution):
        
        model= self.getModelForSolution(solution)
        envs= self.__tsEnvs
        
        
        # Initialize test environments
        active_envs= list(range(self.__nTests))
        R_test= [0]*self.__nTests
        S= []
        if self.__test_seeds:
            self.__testCounter= 1
        for i, env in enumerate(envs):
            s, _= env.reset(seed= self.__testCounter)
            self.__testCounter+= 1
            S.append(s)
        
        # Run environments in pseudo-parallel
        while active_envs:
            
            # Get action
            inputs= [S[i] for i in active_envs]
            #t_inputs= tf.convert_to_tensor(inputs, dtype=tf.float32)
            
            logits= model(inputs)
            actions= self.__action_selector(logits.numpy())
            
            #actions= np.clip(actions, a_min=-1.0, a_max=1.0)
            remove_envs= []
            for action, env_idx in zip(actions, active_envs):
                
                sp, reward, terminated, truncated, _ = envs[env_idx].step(action)
                R_test[env_idx]+= reward
                
                done= terminated or truncated
                if done:
                    remove_envs.append(env_idx)
                else:
                    S[env_idx]= sp
            for env_idx in remove_envs:
                active_envs.remove(env_idx)
                
        fitness= np.mean(R_test)
        if self.__reward_solved is not None and fitness >= self.__reward_solved:
            self.__problem_solved= True
        
        return fitness, 1
    
    
    
    
class ClassificationEvaluator:
    
        
    
    """
    INPUT: dictionary params containing:
        params['X']: Input data
        params['Y']: Output data
        params['model_builder']: Callable to build a classification model
        params['error_solved']: Percentage of error to consider the problem as solved
    """
    def __init__(self, params):
        self.__params= params
        self.reset()


    """
    Resets the evaluator to defaults
    """
    def reset(self):
        params= self.__params

        self.__X= params['X']
        self.__Y= params['Y']
        self.__numClasses= len(np.unique(self.__Y))
        self.__model_builder= params['model_builder']
        self.__error_solved= params['error_solved']
        self.__model_param= params['model_param']
        self.__problem_solved= False
        self.__y_real= self.__Y


    def problem_solved(self):
        if self.__error_solved is None:
            return False
        return self.__problem_solved
        



    def getModelForSolution(self, solution):
        model_param= self.__model_param
        model_param['solution']= solution
        return self.__model_builder(model_param)

    

    def predict(self, solution, x= None):

        model= self.getModelForSolution(solution)
        num_classes= self.__numClasses
        
        e_v= model(self.__X).numpy()
        if num_classes != 2:
            e_v= e_v.reshape(-1, num_classes)
            y_pred= np.argmax(e_v, axis=1)
        else:
            e_v= e_v.reshape(-1)
            y_pred= e_v >= 0
            
        return y_pred.astype(int)



    def __call__(self, solution):
        
        error_solved= self.__error_solved
        num_classes= self.__numClasses
        
        model= self.getModelForSolution(solution)
        
        e_v= model(self.__X).numpy()
        if num_classes != 2:
            e_v= e_v.reshape(-1, num_classes)
            y_pred= np.argmax(e_v, axis=1)
        else:
            e_v= e_v.reshape(-1)
            y_pred= e_v >= 0
        
        perc_error= 100*np.mean( self.__y_real != y_pred )
        
        if error_solved is not None and perc_error <= error_solved:
            self.__problem_solved= True

        return perc_error, 1




class ClusteringEvaluator:
    
        
    
    """
    INPUT: dictionary params containing:
        params['X']: Input data
        params['model_builder']: Callable to build a classification model
        params['target_metric']: Target value to consider the problem is solved
    """
    def __init__(self, params):
        self.__params= params

        self.reset()


    """
    Resets the evaluator to defaults
    """
    def reset(self):
        params= self.__params

        self.__X= params['X']
        if 'clustering_method' in params.keys():
            self.__cluster_method= params['clustering_method']
        else:
            self.__cluster_method= None
        self.__model_builder= params['model_builder']
        self.__target_metric= params['target_metric']
        self.__model_param= params['model_param']
        self.__problem_solved= False



    def problem_solved(self):
        if self.__target_metric is None:
            return False
        return self.__problem_solved

        



    def getModelForSolution(self, solution):
        model_param= self.__model_param
        model_param['solution']= solution
        return self.__model_builder(model_param)

    
    
    def predict(self, solution, x= None):
        if x is None:
            x= self.__X
        model= self.getModelForSolution(solution)
        e_v= model(x).numpy()
        
        if self.__cluster_method is not None:
            y_pred= self.__cluster_method(e_v)
        else:
            if len(e_v.shape)> 1 and e_v.shape[1]>1:
                num_clusters= e_v.shape[1]
            else:
                num_clusters= 2
            if num_clusters != 2:
                e_v= e_v.reshape(-1, num_clusters)
                y_pred= np.argmax(e_v, axis=1)
            else:
                e_v= e_v.reshape(-1)
                y_pred= e_v >= 0

        return y_pred.astype(int)



    def __call__(self, solution):
        
        d= lambda v1,v2: np.sum((v1-v2)**2)
        
        target_metric= self.__target_metric
        
        y_pred= self.predict(solution)
        X= self.__X
        num_clusters= len(np.unique(y_pred))
        
        if num_clusters == 1:
            S= -1
        else:
            
            # Silhouette Score
            S= 0
            for i in range(len(y_pred)):
                
                cluster_i= y_pred[i]
                neighbours= np.where(y_pred==cluster_i)[0]
                a_i= 0
                for j in neighbours:
                    if j != i:
                        a_i+= d(X[i, :], X[j, :])
                if len(neighbours) > 1:
                    a_i/= len(neighbours)-1
                else:
                    a_i= None
                
                b_i= None
                for cluster_j in range(num_clusters):
                    if cluster_j == cluster_i:
                        continue
                    
                    outlanders= np.where(y_pred == cluster_j)[0]
                    b_j= 0                
                    if len(outlanders) > 0:
                        for j in outlanders:
                            b_j+= d(X[i, :], X[j, :])
                        b_j/=len(outlanders)
                        if b_i is None or b_j < b_i:
                            b_i= b_j
                    
                if b_i is None or a_i is None:
                    s_i= -1
                else:
                    s_i= (b_i - a_i)/ np.max([a_i, b_i])
                S+= s_i
            S/= len(y_pred)
        
        if target_metric is not None and S >= target_metric:
            self.__problem_solved= True
        
        return S, 1
