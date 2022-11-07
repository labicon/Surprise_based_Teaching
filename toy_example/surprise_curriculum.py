# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:29:39 2022

@author: clemm
"""

import gym 
import torch 
import numpy as np 
import math 
from collections import deque, namedtuple

'''
initialize environment 
'''
env = gym.make('MountainCarContinuous-v0', render_mode = "human")
observation, infor = env.reset()

'''
training hyperparameters
'''
iters = 10 
batch_size = 32 


Transition = namedtuple("Transition", ("state", "action", "reward", "new_state"))
class TeacherReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.size = size 
    
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.memory = Transition([],[],[],[])
        
    def insert(self, rollouts):
        for j in range(len(rollouts.state)):
            self.memory.state.append(rollouts.state[j])
            self.memory.action.append(rollouts.action[j])
            self.memory.reward.append(rollouts.reward[j])
            self.memory.new_state.append(rollouts.new_state[j])
           
        while len(self.memory.state)> self.size: 
            indx = np.random.choice(len(self.memory.state), size = 1, replace = False)
            indx = int(indx)
            del self.memory.state[indx]
            del self.memory.action[indx]
            del self.memory.reward[indx]
            del self.memory.new_state[indx]
           
    def sample_batch(self, batch_size):
        samples = Transition([], [], [], [])
        indxs = np.random.choice(len(self.memory.state), size = batch_size, replace = False)
        if len(self.memory.state) < self.size: 
            indxs = indxs[0:len(self.memory.state)-1]
        
        for k in range(len(indxs)):
            samples.state.append(self.memory.state[indxs[k]])
            samples.action.append(self.memory.action[indxs[k]])
            samples.reward.append(self.memory.reward[indxs[k]])
            samples.new_state.append(self.memory.new_state[indxs[k]])
            
        return samples
        
def surprisal_reward(teacher_reward, teacher_policy, student_policy): 
    
    surprise_reward = 0 
    return  surprise_reward
 
def rollout(model, env, states, num_steps_per_rollout, epsilon, device):
    rollouts = Transition([], [], [], [])
    with torch.no_grad(): 
        for j in range(num_steps_per_rollout): 
            states = torch.from_numpy(np.array(states)).float().to(device)
            '''
            Get action from policy model
            '''
            action = []
            action = action.numpy()
            step = env.step(action)
            new_states, rewards, dones, infos = list(zip(*step))
            for jj in range(len(new_states)): 
                rollouts.state.append(states[jj])
                rollouts.action.append(action[jj])
                rollouts.reward.append(rewards[jj])
                rollouts.new_state.append(new_states[jj])
 
    return rollouts

def training(env, teacher_model, student_model, iters, batch_size): 
    '''
    initialize training paramaters 
    '''
    
    for i in range(iters): 
        
        
        pass 
    
    return







