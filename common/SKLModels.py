#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:31:29 2024

@author: manupc
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
import numpy as np
import time


"""
Implementation of the MLP Classifier
"""
class SKLMLPClassifier:
    
    """
    params: Dictionary containing:
        params['Xtr']: Input training data
        params['Ytr']: Output training data
        params['Xts']: Input test data
        params['Yts']: Output test data
        params['hidden_layer_sizes']: Tuple containing number of hidden neurons
        params['alpha']: Learning rate
        params['solver']: Solver ('sgd' , 'adam')
        params['parameters']: None, or parameters for estimation (do not fit)
        params['metric']: 'perc_error' / 'perc_accuracy'
    """
    def __init__(self, params):

        xtr= params['Xtr']
        xts= params['Xts']
        ytr= params['Ytr']
        yts= params['Yts']
        metric= params['metric']
        hidden_layer_sizes= params['hidden_layer_sizes']
        alpha= params['alpha']
        solver= params['solver']
        
        
        self.__XTr= xtr
        self.__XTs= xts
        self.__YTr= ytr
        self.__YTs= yts
        self.__metric= metric
        self.__hidden_layer_sizes= hidden_layer_sizes
        self.__alpha= alpha
        self.__solver= solver
        
        if 'parameters' in params.keys() and params['parameters'] is not None:
            self.__parameters= params['parameters']
        else:
            self.__parameters= None

 
    def perc_error(self, y_real, y_pred):
        return 100*np.mean( y_real != y_pred)

    def perc_accuracy(self, y_real, y_pred):
        return 100*np.mean( y_real == y_pred)


    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['evaluations'] -> Number of solution evaluations
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iteration, evaluations, time, value) of mean fitness per iteration
            out['history_best_fitness'] -> History (iteration, evaluations, time, value) of best fitness update
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        solution= self.__parameters
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None)
        
        XTr= self.__XTr
        YTr= self.__YTr
        XTs= self.__XTs
        YTs= self.__YTs
        metric= self.__metric
        hidden_layer_sizes= self.__hidden_layer_sizes
        alpha= self.__alpha
        solver= self.__solver
        
        if metric == 'perc_error':
            metric_func= self.perc_error
        elif metric == 'perc_accuracy':
            metric_func= self.perc_accuracy
        else:
            raise Exception('SKLMLPRegressor.run: metric {} unknown'.format(metric))    
        
        
        
        t0= time.time()
        alg= MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,solver=solver, max_iter=MaxIterations)
        if solution is None:
            alg.fit(XTr, YTr)
        else:
            alg= solution
        tf= time.time()
        t= tf-t0
            
        TRy_pred= alg.predict(XTr) if XTr is not None else None
        TSy_pred= alg.predict(XTs) if XTs is not None else None
        TrFitness= metric_func(YTr, TRy_pred) if YTr is not None else None
        TsFitness= metric_func(YTs, TSy_pred) if YTs is not None else None
        
        iterations= alg.n_iter_
        
        if verbose:
            print(verbose_text_append+' END. It. {}, metric= {:.3f}. test= {:.3f}. t= {:.2f}'.format(iterations, TrFitness, TsFitness, t))
    
        
        best= alg
        
        out= {}
        out['iterations']= iterations
        out['best']= best
        out['best_fitness']= TrFitness
        out['test_performance']= TsFitness
        out['test_predictions']= TSy_pred
        out['train_predictions']= TRy_pred
        out['time']= t

        return out







"""
Implementation of the MLP Regressor
"""
class SKLMLPRegressor:
    
    """
    params: Dictionary containing:
        params['Xtr']: Input training data
        params['Ytr']: Output training data
        params['Xts']: Input test data
        params['Yts']: Output test data
        params['hidden_layer_sizes']: Tuple containing number of hidden neurons
        params['alpha']: Learning rate
        params['solver']: Solver ('sgd' , 'adam')
        params['parameters']: None, or parameters for estimation (do not fit)
        params['metric']: 'MSE' / 'MAE'
    """
    def __init__(self, params):

        xtr= params['Xtr']
        xts= params['Xts']
        ytr= params['Ytr']
        yts= params['Yts']
        metric= params['metric']
        hidden_layer_sizes= params['hidden_layer_sizes']
        alpha= params['alpha']
        solver= params['solver']
        
        
        self.__XTr= xtr
        self.__XTs= xts
        self.__YTr= ytr
        self.__YTs= yts
        self.__metric= metric
        self.__hidden_layer_sizes= hidden_layer_sizes
        self.__alpha= alpha
        self.__solver= solver
        
        if 'parameters' in params.keys() and params['parameters'] is not None:
            self.__parameters= params['parameters']
        else:
            self.__parameters= None

 
    def MSE(self, y_real, y_pred):
        return np.mean( (y_real - y_pred)**2 )

    def MAE(self, y_real, y_pred):
        return np.mean( np.abs(y_real - y_pred) )


    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['evaluations'] -> Number of solution evaluations
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iteration, evaluations, time, value) of mean fitness per iteration
            out['history_best_fitness'] -> History (iteration, evaluations, time, value) of best fitness update
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        solution= self.__parameters
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None)
        
        XTr= self.__XTr
        YTr= self.__YTr
        XTs= self.__XTs
        YTs= self.__YTs
        metric= self.__metric
        hidden_layer_sizes= self.__hidden_layer_sizes
        alpha= self.__alpha
        solver= self.__solver
        
        if metric == 'MSE':
            metric_func= self.MSE
        elif metric == 'MAE':
            metric_func= self.MAE
        else:
            raise Exception('SKLMLPRegressor.run: metric {} unknown'.format(metric))    
        
        
        
        t0= time.time()
        alg= MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,solver=solver, max_iter=MaxIterations)
        if solution is None:
            alg.fit(XTr, YTr)
        else:
            alg= solution
        tf= time.time()
        t= tf-t0
            
        TRy_pred= alg.predict(XTr) if XTr is not None else None
        TSy_pred= alg.predict(XTs) if XTs is not None else None
        TrFitness= metric_func(YTr, TRy_pred) if YTr is not None else None
        TsFitness= metric_func(YTs, TSy_pred) if YTs is not None else None
        
        iterations= alg.n_iter_
        
        if verbose:
            print(verbose_text_append+' END. It. {}, metric= {:.3f}. test= {:.3f}. t= {:.2f}'.format(iterations, TrFitness, TsFitness, t))
    
        
        best= alg
        
        out= {}
        out['iterations']= iterations
        out['best']= best
        out['best_fitness']= TrFitness
        out['test_performance']= TsFitness
        out['test_predictions']= TSy_pred
        out['train_predictions']= TRy_pred
        out['time']= t

        return out










"""
Implementation of the Logistic Regression Classifier
"""
class SKLLogisticRegression:
    
    """
    params: Dictionary containing:
        params['Xtr']: Input training data
        params['Ytr']: Output training data
        params['Xts']: Input test data
        params['Yts']: Output test data
        params['parameters']: None, or parameters for estimation (do not fit)
        params['metric']: 'perc_error' / 'perc_accuracy'
    """
    def __init__(self, params):

        xtr= params['Xtr']
        xts= params['Xts']
        ytr= params['Ytr']
        yts= params['Yts']
        metric= params['metric']
        
        self.__XTr= xtr
        self.__XTs= xts
        self.__YTr= ytr
        self.__YTs= yts
        self.__metric= metric
        if 'parameters' in params.keys() and params['parameters'] is not None:
            self.__parameters= params['parameters']
        else:
            self.__parameters= None

 
    def perc_error(self, y_real, y_pred):
        return 100*np.mean( y_real != y_pred)

    def perc_accuracy(self, y_real, y_pred):
        return 100*np.mean( y_real == y_pred)


    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['evaluations'] -> Number of solution evaluations
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iteration, evaluations, time, value) of mean fitness per iteration
            out['history_best_fitness'] -> History (iteration, evaluations, time, value) of best fitness update
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        solution= self.__parameters
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None)
        
        XTr= self.__XTr
        YTr= self.__YTr
        XTs= self.__XTs
        YTs= self.__YTs
        metric= self.__metric
        
        if metric == 'perc_error':
            metric_func= self.perc_error
        elif metric == 'perc_accuracy':
            metric_func= self.perc_accuracy
        else:
            raise Exception('SKLLogisticRegression.run: metric {} unknown'.format(metric))    
        
        
        
        t0= time.time()
        alg= LogisticRegression(max_iter=MaxIterations)
        if solution is None:
            alg.fit(XTr, YTr)
        else:
            alg= solution
        tf= time.time()
        t= tf-t0
            
        TRy_pred= alg.predict(XTr) if XTr is not None else None
        TSy_pred= alg.predict(XTs) if XTs is not None else None
        TrFitness= metric_func(YTr, TRy_pred) if YTr is not None else None
        TsFitness= metric_func(YTs, TSy_pred) if YTs is not None else None
        
        iterations= alg.n_iter_[0]
        
        if verbose:
            print(verbose_text_append+' END. It. {}, metric= {:.3f}. test= {:.3f}. t= {:.2f}'.format(iterations, TrFitness, TsFitness, t))
    
        
        best= alg
        
        out= {}
        out['iterations']= iterations
        out['best']= best
        out['best_fitness']= TrFitness
        out['test_performance']= TsFitness
        out['test_predictions']= TSy_pred
        out['train_predictions']= TRy_pred
        out['time']= t

        return out





"""
Implementation of the KMeans clustering
"""
class SKLKMeans:
    
    """
    params: Dictionary containing:
        params['Xtr']: Input training data
        params['Xts']: Input test data
        params['n_clusters']: Number of clusters
        params['parameters']: None, or parameters for estimation (do not fit)
        params['metric']: 'silhouette'
    """
    def __init__(self, params):

        xtr= params['Xtr']
        xts= params['Xts']
        metric= params['metric']
        n_clusters= params['n_clusters']
        
        
        self.__XTr= xtr
        self.__XTs= xts
        self.__metric= metric
        self.__n_clusters= n_clusters
        
        if 'parameters' in params.keys() and params['parameters'] is not None:
            self.__parameters= params['parameters']
        else:
            self.__parameters= None

 
    def silhouette(self, X, y_pred):
        d= lambda v1,v2: np.sum((v1-v2)**2)
        num_clusters= len(np.unique(y_pred))
    
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
        return S




    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['evaluations'] -> Number of solution evaluations
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iteration, evaluations, time, value) of mean fitness per iteration
            out['history_best_fitness'] -> History (iteration, evaluations, time, value) of best fitness update
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        solution= self.__parameters
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None)
        
        XTr= self.__XTr
        XTs= self.__XTs
        metric= self.__metric
        n_clusters= self.__n_clusters
        
        if metric == 'silhouette':
            metric_func= self.silhouette
        else:
            raise Exception('SKLKMeans.run: metric {} unknown'.format(metric))    
        
        
        
        t0= time.time()
        alg= KMeans(n_clusters= n_clusters, max_iter=MaxIterations, n_init='auto', init='random')
        if solution is None:
            alg.fit(XTr)
        else:
            alg= solution
        tf= time.time()
        t= tf-t0
            
        TRy_pred= alg.predict(XTr) if XTr is not None else None
        TSy_pred= alg.predict(XTs) if XTs is not None else None
        TrFitness= metric_func(XTr, TRy_pred) if XTr is not None else None
        TsFitness= metric_func(XTs, TSy_pred) if XTs is not None else None
        
        iterations= alg.n_iter_
        
        if verbose:
            print(verbose_text_append+' END. It. {}, metric= {:.3f}. test= {:.3f}. t= {:.2f}'.format(iterations, TrFitness, TsFitness, t))
    
        
        best= alg
        
        out= {}
        out['iterations']= iterations
        out['best']= best
        out['best_fitness']= TrFitness
        out['test_performance']= TsFitness
        out['test_predictions']= TSy_pred
        out['train_predictions']= TRy_pred
        out['time']= t

        return out


