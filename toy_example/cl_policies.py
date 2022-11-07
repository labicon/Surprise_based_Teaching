# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class TeacherModel(): 
    def __init__(self, input_dim, output_dim, hidden_layers): 
        super(TeacherModel, self).__init__()
        # 1 hidden layer of some dimension -- probably need to change ? 
        self.layers =nn.Sequential(nn.Linear(input_dim, hidden_layers ), 
                              nn.Tanh(), 
                              nn.Linear(hidden_layers,hidden_layers +1  ), 
                              nn.Tanh())
        self.actor = nn.Linear(hidden_layers +1, output_dim)
        self.critic = nn.Linear(hidden_layers +1, 1 )
        return
    
    def forward(self, x): 
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        
        return actor, critic 
    
    
    
class StudentModel(): 
    def __init__(self): 
        super(StudentModel, self).__init__()
        
        return 
    
    def forward(self, x): 
        return 
