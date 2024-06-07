#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:56:47 2024

@author: manupc
"""

import gymnasium as gym

env= gym.make('FrozenLake-v1', is_slippery=True)

OptimalPolicy= [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

CumR= 0
NEP= 100

for ep in range(NEP):
    
    R= 0
    s, _= env.reset()
    done= False
    while not done:
        
        s, r, tru, ter, _= env.step(OptimalPolicy[s])
        done= tru or ter
        R+= r
    CumR+= R
AvgR= CumR/NEP
print(AvgR)