#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:02:13 2022

@author: w044elc
"""
import torch.nn as nn
import numpy as np 
from garage.torch.policies import GaussianMLPPolicy

class MLP(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        self.layers = nn.Sequential(nn.Linear(input_dim, 64), 
                                    nn.ReLU(), 
                                    nn.Linear(64, 32), 
                                    nn.ReLU(), 
                                    nn.Linear(32, output_dim))
        return 
    
    def forward(self, x): 
        out = self.layers(x)
        return out 
    
    
    
class GaussianMLPRegression():
    def __init__(self, input_shape, output_shape, env): 
        
        self.network = MLP(input_shape, output_shape)
                                         
        self.optimizer = None
        self.step = .01 
        
        return 
