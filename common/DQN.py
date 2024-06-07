#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:46:27 2024

@author: manuel
"""

from collections import deque
import time
import numpy as np
import tensorflow as tf



################################################################################
################################################################################
## EPSILON-GREEDY EXPLORATION
################################################################################
################################################################################

"""
Exploration policy e-Greedy
"""
class EpsilonGreedyExploration:

    UPDATE_EPISODES= 0    
    UPDATE_ITERATIONS= 1
    
    DECREASE_LINEAR= 0

    """
    params: Dictionary containing:
        params['eps0']: Initial epsilon for e-Greedy policy 
        params['epsf']: Final epsilon for e-Greedy policy 
        params['decrease_type']: 'linear' for linear eps decrease
        params['update_type']: e-Greedy update with 'iterations' or 'episodes'
        params['eps_steps']: e-Greedy steps to reach eg_epsf from eg_eps0
    """
    def __init__(self, params):
        eps0= params['eps0']
        epsf= params['epsf']
        update= params['update_type']
        steps= params['eps_steps']
        decrease= params['decrease_type']
        
        self.__eps0= eps0
        self.__epsf= epsf
        if update == 'episodes':
            self.__update_type= EpsilonGreedyExploration.UPDATE_EPISODES 
        elif update == 'iterations':
            self.__update_type= EpsilonGreedyExploration.UPDATE_ITERATIONS
        else:
            raise Exception('EpsilonGreedyExploration.__init__: Unknown update type {}'.format(update))
        self.__steps= steps
        
        if decrease == 'linear':
            self.__decrease= EpsilonGreedyExploration.DECREASE_LINEAR
        else:
            raise Exception('EpsilonGreedyExploration.__init__: Unknown epsilon decrease type {}'.format(decrease))
        
        self.reset()
    
    
    """
    Resets the exploration policy to initial values
    """
    def reset(self):
        self.__step_counter= 0
        self.__eps= self.__eps0
        self.__last_step= -1
        
        
        
    def getAction(self, model, input_data):
        n_actions= model.outputs[0].shape[-1]
        actions= np.argmax(model(input_data).numpy(), axis=-1).reshape(-1)
        exploration_idx= np.random.rand(len(actions)) <= self.__eps
        n_explore= np.sum(exploration_idx)
        if n_explore == 0: 
            return actions
        
        eg_actions= np.random.randint(low=0, high=n_actions, size=(n_explore))
        actions[exploration_idx]= eg_actions
        return actions
        
        
        
    
    def update(self, iterations, episodes):
        step= iterations if self.__update_type == EpsilonGreedyExploration.UPDATE_ITERATIONS else episodes
        if self.__last_step == step:
            return
        
        self.__last_step= step
        
        if self.__decrease == EpsilonGreedyExploration.DECREASE_LINEAR:
            self.__eps= max(self.__epsf, self.__eps0+step*(self.__epsf-self.__eps0)/self.__steps)
        else:
            raise NotImplementedError()




################################################################################
################################################################################
## REPLAY BUFFERS
################################################################################
################################################################################


"""
Deque replay buffer 
"""
class DequeReplayBuffer:
    """
    params: Dictionary containing:
        params['capacity']: Size of the buffer
        params['populate']: True to populate the buffer initially, False otherwise
    """
    def __init__(self, params):
        self.__capacity= params['capacity']
        self.__require_populate= params['populate']
        self.__buffer= deque(maxlen=self.__capacity)
        
    
    def require_populate(self):
        return self.__require_populate
    
    def capacity(self):
        return self.__capacity
    
    def clear(self):
        self.__buffer.clear()
        
        
    def __len__(self):
        return len(self.__buffer)


    def append(self, experience):
        self.__buffer.append(experience)

    def sample(self, batch_size):
        indices= np.random.choice(len(self.__buffer), size=batch_size, replace=False)
        return zip(*[self.__buffer[idx] for idx in indices])



################################################################################
################################################################################
## TARGET UPDATE STRATEGIES
################################################################################
################################################################################

"""
hard (full copy) target network update
"""
class HardUpdate:
    
    UPDATE_EPISODES= 0    
    UPDATE_ITERATIONS= 1


    """
    params: Dictionary containing:
        params['iterations']: Number of iterations before hard target synchronization
        params['update_type']: Hard update with 'iterations' or 'episodes'
        params['steps']: Steps to perform update

    """
    def __init__(self, params):

        update= params['update_type']
        if update == 'episodes':
            self.__update_type= HardUpdate.UPDATE_EPISODES 
        elif update == 'iterations':
            self.__update_type= HardUpdate.UPDATE_ITERATIONS
        else:
            raise Exception('EpsilonGreedyExploration.__init__: Unknown update type {}'.format(update))
        self.__steps= params['steps']

        self.reset()
        
    def reset(self):
        self.__step_counter= 0
        self.__last_step= -1
    
    def copy(self, target, model):
        target.set_weights(model.get_weights())
    
    def sync(self, target, model, iterations, episodes):

        step= iterations if self.__update_type == EpsilonGreedyExploration.UPDATE_ITERATIONS else episodes
        if self.__last_step == step:
            return
        
        self.__last_step= step
        
        self.__step_counter+= 1
        if self.__step_counter < self.__steps:
            return
        self.__step_counter= 0
        target.set_weights(model.get_weights())
        






"""
soft target network update
"""
class SoftUpdate:
    
    UPDATE_EPISODES= 0    
    UPDATE_ITERATIONS= 1
    
    
    """
    params: Dictionary containing:
        params['alpha']: Rate of target soft update as(1-alpha)*target + self.alpha*model

    """
    def __init__(self, params):
        self.__alpha= params['alpha']
        self.reset()
        
    def reset(self):
        pass
    
    def copy(self, target, model):
        target.set_weights(model.get_weights())
    
    
    def sync(self, target, model, iterations, episodes):
        alpha= self.__alpha
        weights= []
        for t_w, m_w in zip(target.get_weights(), model.get_weights()):
            w_i= (1-alpha)*t_w + alpha*m_w
            weights.append(w_i)
        target.set_weights(weights)
        




"""
Implementation of the REINFORCE algorithm with RTG and Entropy Bonus
"""
class DQN:
    
    
    TEST_EPISODES= 0    
    TEST_ITERATIONS= 1
    
    """
    params: Dictionary containing:
        params['env_builder']: Callable to build the environment
        params['gamma']: Discount factor
        params['model_builder']: callable to create a policy that returns logits
        params['optimizer']: Optimization algorithm
        params['batch_size']: Batch Size
        params['training_envs']: Number of environments to populate buffer (default= 1)
        params['buffer']['type']: Type of buffer ('deque')
        params['buffer'][...]: Parameters of the buffer type (e.g. DequeReplayBuffer class)
        params['exploration']['type']: Type of exploration (default='eGreedy')
        params['exploration'][...]: Parameters of the specific exploration policy (e.g. EpsilonGreedyExploration class)
        params['test']['reference']: 'iterations' or 'episodes' (or None)
        params['test']['steps']: None, or int to set the number of algorithm iterations/episodes to wait before testing
        params['test']['size']: Number of test episodes 
        params['test']['RewardSolved']: Reward value to consider environment solved, or None
        params['target_update']['type']: 'hard' for hard update/'soft' for soft update
        params['target_update'][...]: Parameters of the specific target update (e.g. HardUpdate class)
        params['DoubleDQN']: True to activate DoubleDQN rule, False to set usual DQN
        params['mean_history_horizon']: Number of past episodes to consider in the history of mean training returns
    """
    def __init__(self, params):

        env_builder= params['env_builder']
        gamma= params['gamma']
        doubleDQN= params['DoubleDQN']
        model_builder= params['model_builder']
        optimizer= params['optimizer']
        BS= params['batch_size']
        if 'training_envs' not in params:
            training_envs= 1
        else:
            training_envs= params['training_envs']
        
        buffer_type= params['buffer']['type']
        if buffer_type == 'deque':
            buffer= DequeReplayBuffer(params['buffer'])
        else:
            raise Exception('Buffer type {} not implemented'.format(buffer_type))

        exploration_type= params['exploration']['type']
        if exploration_type == 'eGreedy':
            exploration_policy= EpsilonGreedyExploration(params['exploration'])
        else:
            raise Exception('Exploration type {} not implemented'.format(exploration_type))
        
        target_update_type= params['target_update']['type']
        if target_update_type == 'hard':
            target_update= HardUpdate(params['target_update'])
        elif target_update_type == 'soft':
            target_update= SoftUpdate(params['target_update'])
        else:
            raise Exception('Target Update type {} not implemented'.format(target_update_type))
            
        
        
        
        test_reference= params['test']['reference']
        if test_reference == 'iterations':
            test_reference= DQN.TEST_ITERATIONS
        elif test_reference == 'episodes':
            test_reference= DQN.TEST_EPISODES
        else:
            raise Exception('Test Reference type {} not recognized'.format(test_reference))
        test_steps= params['test']['steps']
        if test_steps is not None:
            test_envs= params['test']['size']
        else:
            test_envs= 0

        reward_solved= params['test']['reward_solved']
        mean_history_horizon= params['mean_history_horizon']
        
        self.__return_history= deque(maxlen=mean_history_horizon)

        self.__env_builder= env_builder
        self.__gamma= gamma
        self.__doubleDQN= doubleDQN
        self.__model_builder= model_builder
        self.__optimizer= optimizer
        self.__batch_size= BS
        self.__buffer= buffer
        self.__exploration_policy= exploration_policy
        self.__target_update= target_update
        
        self.__trEnvs= [env_builder() for _ in range(training_envs)]
        
        self.__tsEnvs= [env_builder() for _ in range(test_envs)]
        self.__test_reference= test_reference
        self.__test_steps= test_steps
        self.__reward_solved= reward_solved
        
        

       
    def __update_stopping_criterion(self):
        if (self.__MaxIterations is not None and self.__it >= self.__MaxIterations) or\
            (self.__MaxEpisodes is not None and self.__episodes >= self.__MaxEpisodes) or\
            self.__problem_solved:
            self.__stopping_criterion= True
        return self.__stopping_criterion
    



    def __test_model(self, model):
        
        tsEnvs= self.__tsEnvs
        BS= len(tsEnvs)
        active_envs= list(range(BS))
        
        
        # Reset envs
        R= [0]*BS
        S= []
        for env in tsEnvs:
            s, _= env.reset()
            S.append(s)
        
        # Run environments in pseudo-parallel
        while active_envs:
            
            # Get action
            inputs= [S[i] for i in active_envs]

            t_inputs= tf.convert_to_tensor(inputs, dtype=tf.float32)
            logits= model(t_inputs)
            actions= tf.argmax(input= logits, axis=1).numpy().reshape(-1)
            
            remove_envs= []
            for action, env_idx in zip(actions, active_envs):
                sp, reward, terminated, truncated, _ = tsEnvs[env_idx].step(action)
                R[env_idx]+= reward
                done= terminated or truncated
                if done:
                    remove_envs.append(env_idx)
                else:
                    S[env_idx]= sp
            for env_idx in remove_envs:
                active_envs.remove(env_idx)
        return np.mean(R)

        
        
    def __env_step(self, model, target):
        
        S= self.__S
        Rt= self.__Rt

        trEnvs= self.__trEnvs
        exploration_policy= self.__exploration_policy
        target_update= self.__target_update
        buffer= self.__buffer
        return_history= self.__return_history
        reward_solved= self.__reward_solved
        tests= len(self.__tsEnvs)
        
        input_data= tf.convert_to_tensor(S, dtype=tf.float32)
        actions= exploration_policy.getAction(model, input_data)
        for i, env in enumerate(trEnvs):
            a= actions[i]
            s= S[i]
            sp, r, terminated, truncated, _ = env.step(a)
            done= terminated or truncated
            buffer.append( (s, a, r, sp, done) )
            Rt[i]+= r
            S[i]= sp
            if done:
                self.__episodes+= 1
                t= time.time() - self.__t0
                currentR= Rt[i]
                history_record= ( self.__it, self.__episodes, t, currentR)
                self.__AvgScoreH.append(history_record)
                
                
                exploration_policy.update(iterations= self.__it, episodes= self.__episodes)
                target_update.sync(target= target, model= model, iterations= self.__it, episodes= self.__episodes)
                return_history.append(Rt[i])
                s, _= env.reset()
                S[i]= s
                Rt[i]= 0
                if reward_solved is not None and\
                   tests == 0 and\
                   np.mean(return_history)>= reward_solved:
                       self.__problem_solved= True
                       return
                       

    def __update_best(self, model, fitness):
        self.__target_update.copy(self.__best, model)
        self.__best_fitness= fitness


    def __train(self, model, target):

        buffer= self.__buffer
        batch_size= self.__batch_size
        doubleDQN= self.__doubleDQN
        gamma= self.__gamma
        optimizer= self.__optimizer
        
        # Fetch batch
        bS, bA, bR, bSp, bD = buffer.sample(batch_size)
        bS= np.array(bS, copy=False, dtype=np.float32)
        bA= np.array(bA, copy=False, dtype=int).reshape(-1, 1)
        bR= np.array(bR, copy=False, dtype=np.float32)
        bSp= np.array(bSp, copy=False, dtype=np.float32)
        bD= np.array(bD, copy=False, dtype=bool)

        tS= tf.convert_to_tensor(bS)
        tA= tf.reshape(tf.convert_to_tensor(bA), shape=(-1,1))
        tSp= tf.convert_to_tensor(bSp)


        #  Qtarget(s,a)= r + gamma*max_{a'}Q(s',a')
        if doubleDQN:
            tAp= tf.reshape(tf.argmax(model(tSp), axis=-1), shape=(-1, 1))
            Qvals= tf.gather_nd(target(tSp).numpy(), tAp, batch_dims=1)
        else:
            Qvals= tf.reduce_max(target(tSp), axis=-1)
        Qtarget= np.where(bD, bR, bR+gamma*Qvals)
        Qtarget= tf.convert_to_tensor(Qtarget, dtype=tf.float32)

        with tf.GradientTape() as tape:
            Qnet= tf.gather_nd(model(tS), tA, batch_dims=1)
            loss = tf.math.reduce_mean(tf.square(Qnet - Qtarget))

            variables = model.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
        return float(loss.numpy())

                
                
                



    """
    Runs the algorithm
    INPUTS:
        dictionary params containing:
        params['MaxIterations']: Maximum number of algorithm's iterations
        params['MaxEpisodes']: Approximated Maximum number of sepisodes
        params['verbose']: True to show results in Console
        params['verbose_text_append']: Text to append in verbose mode True
    OUTPUTS:
        dictionary out containing:
            out['iterations'] -> Number of algorithm's iterations
            out['episodes'] -> Number of episodes
            out['best'] -> Best solution found
            out['best_fitness'] -> Fitness of best solution
            out['time'] -> Computational time in s.
            out['history_mean_fitness'] -> History (iteration, evaluations, time, value) of mean fitness per iteration in last 100 episodes
            out['history_best_fitness'] -> History (iteration, evaluations, time, value) of best fitness update in test
            out['history_loss'] -> History(iteration, evaluations, time, value) of training loss
    """
    def run(self, params):
        
        MaxIterations= params['MaxIterations']
        MaxEpisodes= params['MaxEpisodes']
        verbose= params['verbose']
        verbose_text_append= params['verbose_text_append']
        assert(MaxIterations is not None or MaxEpisodes is not None)
        
        
        self.__best= self.__model_builder()
        self.__best_fitness= -np.inf
        self.__stopping_criterion= False
        self.__problem_solved= False
        self.__MaxIterations= MaxIterations
        self.__MaxEpisodes= MaxEpisodes

        self.__AvgScoreH= []
        BestScoreH= []
        LossH= []

        
        # Initialization
        trEnvs= self.__trEnvs
            
        reward_solved= self.__reward_solved
        return_history= self.__return_history
        buffer= self.__buffer
        batch_size= self.__batch_size
        exploration_policy= self.__exploration_policy
        target_synchronizer= self.__target_update
        model= self.__model_builder()
        target= self.__model_builder()

        return_history.clear()
        buffer.clear()
        exploration_policy.reset()
        target_synchronizer.reset()
        target_synchronizer.copy(target= target, model= model)

        test_steps= self.__test_steps
        test_reference= self.__test_reference

        self.__S= []
        self.__Rt= []
        
        # Reset training envs
        for i, env in enumerate(trEnvs):
            s, _= env.reset()
            self.__S.append(s)
            self.__Rt.append(0)


        self.__t0= time.time()


        # Populate buffer
        buffer_min_size= buffer.capacity() if buffer.require_populate() else batch_size 
        env= self.__env_builder()
        s, _= env.reset()
        R= 0
        while len(buffer) < buffer_min_size:
            
            input_data= tf.reshape(tf.convert_to_tensor(s, dtype=tf.float32), shape=(1, -1))

            a= exploration_policy.getAction(model, input_data)[0]
            sp, r, terminated, truncated, _ = env.step(a)
            R+= r
            done= terminated or truncated
            buffer.append( (s, a, r, sp, done) )
            if not done:
                s= sp
            else:
                return_history.append(R)
                R= 0
                s, _= env.reset()
        
        self.__episodes= 0
        self.__it= 0
        
        
        
        last_episode_show= -1
        last_test_step= -test_steps
        while not self.__update_stopping_criterion():
            
            # Step in the training environments
            self.__env_step(model, target)


            # training
            current_loss= self.__train(model, target)


            
            self.__it+= 1
            exploration_policy.update(iterations= self.__it, episodes= self.__episodes)
            target_synchronizer.sync(target= target, model= model, iterations= self.__it, episodes= self.__episodes)
            

            #Logging
            t= time.time()-self.__t0
            meanR= np.mean(return_history)
            loss_record= ( self.__it, self.__episodes, t, current_loss)
            LossH.append(loss_record)

            
            # Check if test is required
            R_test= None
            current_test_step= self.__it if test_reference == DQN.TEST_ITERATIONS else self.__episodes
            if test_steps is not None:
                if current_test_step >= last_test_step + test_steps:
                    last_test_step= current_test_step
                    R_test= self.__test_model(model)

    
            # Check if environment solved
            if reward_solved is not None:
            
                if test_steps is not None:
                    fitness= R_test
                else:
                    fitness= meanR
                if fitness is not None:
                    if fitness >= reward_solved:
                        self.__problem_solved= True
                    self.__update_best(model, R_test)
                    history_record= ( self.__it, self.__episodes, t, fitness)
                    BestScoreH.append(history_record)
            else:
                history_record= ( self.__it, self.__episodes, t, meanR)
                BestScoreH.append(history_record)
                self.__update_best(model, meanR)



            #Logging            
            if verbose and last_episode_show != self.__episodes:
                last_episode_show= self.__episodes
                best_fitness= self.__best_fitness
                if best_fitness is None:
                    best_fitness= -np.inf
                if len(self.__AvgScoreH) > 0:
                    print(verbose_text_append+' It. {}. Epi.= {}. LastR= {:.3f}. AvgR= {:.3f}, Loss= {:.4f}, TestR= {:.3f}. t= {:.3f}'.format(self.__it, self.__episodes, self.__AvgScoreH[-1][-1], np.mean(return_history), LossH[-1][-1], best_fitness, t))
    
        # Final Log
        t= time.time()-self.__t0
        
        
        best_fitness= self.__best_fitness
        if verbose:
            print(verbose_text_append+' END. It. {}. Epi.= {}. LastR= {:.3f}. MeanR= {:.3f}. TestR= {:.3f}. t= {:.3f}'.format(self.__it, self.__episodes, self.__AvgScoreH[-1][-1], np.mean(return_history), best_fitness, t))
        
        out= {}
        out['iterations']= self.__it
        out['episodes']= self.__episodes
        out['best']= self.__best
        out['best_fitness']= self.__best_fitness
        out['time']= t
        out['history_mean_fitness']= self.__AvgScoreH
        out['history_best_fitness']= BestScoreH
        out['history_loss']= LossH

        return out

