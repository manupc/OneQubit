#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:31:29 2024

@author: manupc
"""

import numpy as np
import time


"""
Implementation of the ES(mu ,/+ lambda)
"""
class ES:
    
    """
    params: Dictionary containing:
        params['mu']: Number of parents
        params['lmbda']: Number of children
        params['sol_size']: Size of a solution
        params['sol_evaluator']: Callable to evaluate a solution
        params['lr']: Learning Rate
        params['min_bound_value']: Minimum allowed value for model parameters (None if no minimum value is considered)
        params['max_bound_value']: Maximum allowed value for model parameters (None if no maximum value is considered)
        params['es_plus']: True para ES(mu+lambda), False para ES(mu, lambda)
        params['initialize_method']: 'U' for Uniform distribution, 'N' for normal
        params['maximization_problem']: True for maximization, False for minimization
    """
    def __init__(self, params):

        mu= params['mu']
        lmbda= params['lmbda']
        sol_size= params['sol_size']
        sol_evaluator= params['sol_evaluator']
        lr= params['lr']
        min_bound_value= params['min_bound_value']
        max_bound_value= params['max_bound_value']
        es_plus= params['es_plus']
        initialize_method= params['initialize_method']
        maximization_problem= params['maximization_problem']

        
        
        self.__mu= mu
        self.__lmbda= lmbda
        assert(lmbda >= mu)
        self.__n_children= lmbda//mu
        self.__lr= lr
        assert(lr > 0)
        self.__es_type_plus= es_plus
        
        self.__sol_size= sol_size
        self.__evaluator= sol_evaluator
        self.__min_bounds= min_bound_value
        self.__max_bounds= max_bound_value
        self.__maximization_problem= maximization_problem
        self.__betterThan= self.__greaterThan if self.__maximization_problem else self.__lessThan

        self.__initialize_method= initialize_method
        assert(initialize_method == 'U' or initialize_method == 'N')


    """
    Initializes the population
    method: 'U' for Uniform distribution, 'N' for normal
    """
    def __Initialize_Population(self, method= 'U'):
        
        pop= []
        for _ in range(self.__mu):
            if method == 'U':
                solution= np.random.rand(self.__sol_size)*np.pi
            elif method == 'N':
                solution= np.random.randn(self.__sol_size)
            else:
                raise Exception('Initialization method {} not in ("U"/"N")'.format(method))
            
            if self.__min_bounds is not None and self.__max_bounds is not None:
                
                solution= solution*(self.__max_bounds - self.__min_bounds) + self.__min_bounds
            elif self.__max_bounds is not None:
                solution= solution*self.__max_bounds
            elif self.__min_bounds is not None:
                solution= solution + self.__min_bounds

            pop.append(solution)        
        return pop

        
        
    def __lessThan(self, f1, f2):
        return f1 < f2
    

    def __greaterThan(self, f1, f2):
        return f1 > f2

       
    def __update_stopping_criterion(self):
        if (self.__MaxIterations is not None and self.__it > self.__MaxIterations) or\
            (self.__MaxEvaluations is not None and self.__evaluations >= self.__MaxEvaluations) or\
            self.__problem_solved:
            self.__stopping_criterion= True
        return self.__stopping_criterion
    
       
    
    def __evaluate_solution(self, solution):
        fitness, evals= self.__evaluator(solution)
        self.__evaluations+= evals
        self.__update_best(solution, fitness)
        if self.__evaluator.problem_solved():
            self.__problem_solved= True
        
        return fitness
    
    def __evaluate_population(self, pop):
        scores= []
        for solution in pop:
            scores.append( self.__evaluate_solution(solution))
            if self.__update_stopping_criterion():
                break
        return scores
    
    
    def __update_best(self, solution, fitness):
        copy= self.__best_fitness is None or self.__betterThan(fitness, self.__best_fitness)
        if copy:
            self.__best= solution.copy()
            self.__best_fitness= fitness
            current_time= time.time()-self.__t0
            history_record= ( self.__it, self.__evaluations, current_time, self.__best_fitness)
            self.__BestScoreH.append( history_record )
        return copy
    
 
     
 

       
    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['MaxEvaluations']: Approximated Maximum number of solution's evaluations
        params['verbose']: True to show results in Console
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
        MaxEvaluations= params['MaxEvaluations']
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None or MaxEvaluations is not None)
        
        self.__evaluator.reset()
        
        self.__stopping_criterion= False
        self.__problem_solved= False
        self.__MaxIterations= MaxIterations
        self.__MaxEvaluations= MaxEvaluations
        self.__best, self.__best_fitness = None, None

        self.__AvgScoreH= []
        self.__BestScoreH= []

        
        # Initialization
        self.__evaluations= 0
        self.__it= 0
        
        self.__t0= time.time()
        pop= self.__Initialize_Population(method= self.__initialize_method)
        scores= self.__evaluate_population(pop)
        
        current_time= time.time()-self.__t0
        
        history_record= ( self.__it, self.__evaluations, current_time, np.mean(scores))
        self.__AvgScoreH.append( history_record )
        if verbose:
            print(verbose_text_append+' BEGIN. It. {}, Eval {}. Mean {:.3f}. Best {:.3f}. t= {:.2f}'.format(self.__it, self.__evaluations, self.__AvgScoreH[-1][-1], self.__best_fitness, current_time))
        
        
        while not self.__update_stopping_criterion():
            
            self.__it+= 1
            
            
            # Generate children
            offspring= []
            for i in range(len(pop)):
                
                # Offspring
                for _ in range(self.__n_children):
                    child= pop[i] + np.random.randn(self.__sol_size)*self.__lr
                    if self.__min_bounds is not None or self.__max_bounds is not None:
                        child= np.clip(child, a_min= self.__min_bounds, a_max= self.__max_bounds)
                    offspring.append(child)

            # Population evaluation
            new_scores= self.__evaluate_population(offspring)
            
            # Append parents if ES(mu+lambda)
            if self.__es_type_plus:
                offspring.extend(pop)
                new_scores.extend(scores)

                            

            # Rank solutions
            if self.__maximization_problem:
                ranks= np.argsort( np.argsort(new_scores)[::-1] )
            else:
                ranks= np.argsort( np.argsort(new_scores) )
            selected = [i for i,_ in enumerate(ranks) if ranks[i] < self.__mu]
            pop= []
            scores= []
            for s in selected:
                pop.append(offspring[s])
                scores.append(new_scores[s])
            
            #Logging            
            t= time.time()-self.__t0
            history_record= ( self.__it, self.__evaluations, t, np.mean(scores))
            self.__AvgScoreH.append(history_record)


            # Check secondary stopping criterion in the inner loop
            if self.__update_stopping_criterion():
                break
        
            #Logging            
            if verbose:
                print(verbose_text_append+' It. {}. Eval.= {}. AvgR= {:.3f}, Best= {:.3f}. t= {:.3f}'.format(self.__it, self.__evaluations, self.__AvgScoreH[-1][-1], self.__best_fitness, t))
    
        # Final Log
        t= time.time()-self.__t0
        if len(scores) > 0:
            history_record= ( self.__it, self.__evaluations, t, np.mean(scores))
            self.__AvgScoreH.append( history_record)
        if verbose:
            print(verbose_text_append+' END. It. {}. Eval.= {}. Best= {:.3f}. t= {:.3f}'.format(self.__it, self.__evaluations, self.__best_fitness, t))
        
        out= {}
        out['iterations']= self.__it
        out['evaluations']= self.__evaluations
        out['best']= self.__best
        out['best_fitness']= self.__best_fitness
        out['time']= t
        out['history_mean_fitness']= self.__AvgScoreH
        out['history_best_fitness']= self.__BestScoreH

        return out

