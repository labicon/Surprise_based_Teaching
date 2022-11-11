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
from torch.distributions.categorical import Categorical 
import cl_policies
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
learn_rate = .001
gamma = .95
reset_every = 50
eta0 = .5
Transition = namedtuple("Transition", ("state", "action", "reward", "new_state", "log_prob"))
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
            self.memory.log_prob.append(rollouts.log_prob[j])

        while len(self.memory.state)> self.size: 
            #indx = np.random.choice(len(self.memory.state), size = 1, replace = False)
            #indx = int(indx)
            del self.memory.state[0]
            del self.memory.action[0]
            del self.memory.reward[0]
            del self.memory.new_state[0]
            del self.memory.log_prob[0]

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
            samples.new_state.append(self.memory.log_prob[indxs[k]])

        return samples
        
def surprisal_reward(teacher_reward, actions, actor, value, student_reward, student_policy, eta0): 
    eta1 = eta0/(max(1, abs((1/teacher_reward.size)*np.sum(teacher_reward))))
    eta2 = eta0/(max(1, (1/student_reward.size)*np.sum(student_reward)))
    
    t_dist = Categorical(logits=actor.unsqueeze(-2))
    t_prob = t_dist.log_prob(actions)
    
    #need student log -prob
    s_prob = 0 
    surprise_reward = teacher_reward + eta1*t_prob + eta2*s_prob
    return  surprise_reward
 
def rollout(model, env, states, num_steps_per_rollout, device):
    rollouts = Transition([], [], [], [], [])
    with torch.no_grad(): 
        for j in range(num_steps_per_rollout): 
            states = torch.from_numpy(np.array(states)).float().to(device)
            '''
            Get action from policy model
            '''
            actor, critic = model(states)
            action_dist = Categorical(logits=actor.unsqueeze(-2))
            action = action_dist.probs.argmax(-1)
            log_prob = action_dist.log_prob(action)
            action = action.numpy()
            step = env.step(action)
            new_states, rewards, dones, infos = list(zip(*step))
            for jj in range(len(new_states)): 
                rollouts.state.append(states[jj])
                rollouts.action.append(action[jj])
                rollouts.reward.append(rewards[jj])
                rollouts.new_state.append(new_states[jj])
                rollouts.log_prob.append(log_prob[jj])

            states = new_states
    return rollouts, states

def training(env, teacher_model, student_model, iters, batch_size, learn_rate, gamma,eta0, reset_every): 
    '''
    initialize training paramaters 
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model = cl_policies.TeacherModel()
    student_model= cl_policies.StudentModel()
    
    teacher_a_optim = torch.optim.Adam(teacher_model.parameters(), lr = learn_rate)
    teacher_c_optim = torch.optim.Adam(teacher_model.parameters(), lr = learn_rate)
    student_optim = torch.optim.Adam(student_model.parameters(), lr = learn_rate )
    
    teacher_iters = 10
    student_iters = 10
    
    student_reward = 0 
    
    states = env.reset()
    
    state_dim = env.observation_space.shape[0]
    action_dim= env.action_space.shape[0]
    teacher_memory = TeacherReplayBuffer(100, state_dim, action_dim)
    
    reset_count = 0
    for i in range(iters):
        #put model in train mode
        teacher_model.train()
        #collect teacher rollouts 
        rollouts, states = rollout(teacher_model, env, states, 10, device)
        teacher_memory.insert(rollouts)
        cl_func = torch.nn.MSELoss()
        
        states = torch.vstack(rollouts.state)
        new_states = torch.from_numpy(np.asarray(rollouts.new_state)).float()
        actions = torch.from_numpy(np.asarray(rollouts.action))
        rewards = torch.from_numpy(np.asarray(rollouts.reward)).float()
        probs =  torch.from_numpy(np.asarray(rollouts.log_prob))

        actor, value = teacher_model(states)
        
        rew_sup = suprisal_reward(rewards, actions, actor, value, student_reward, student_model)        
        A = rew_sup - value.detach()
        #normalize advantage ? 
        A = (A - A.mean()) / (A.std()+1e-10) 
        
        critic_loss = nn.MSELoss()

        for j in range(teacher_iters):             
            
            actor, value = teacher_model(states)
            action_dist = Categorical(logits=actor.unsqueeze(-2))
            action = action_dist.probs.argmax(-1)
            curr_prob = action_dist.log_prob(action)
            
            
            ratio = torch.exp(curr_prob - probs)
            
            sur1 = ratio*A
            sur2 = torch.clamp(ratio, .8, 1.2)*A 
            
            al = -1*torch,min(sur1, sur2).mean()
            cl = critic_loss(value, rew_sup)
            
            teacher_a_optim.zero_grad()
            al.backward(retain_graph = True)
            teacher_a_optim.step()
            
            teacher_c_optim.zero_grad()
            cl.backard()
            teacher_c_optim.step
        
            
            
        for j in range(student_iters): 
            batch = teacher_memory.sample_batch(batch_size)
            
            
        #Update student rewards 
        #optimize 
        #step 
        #loss 
        
        
        #validate every %% iterations ? 
        reset_count += 1 
        if reset_count == reset_every: 
            states = env.reset()
            
        pass 
    
    return
