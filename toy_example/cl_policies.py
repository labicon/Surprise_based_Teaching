# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "new_state", "log_prob"))
class TeacherModel(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_layers): 
        super(TeacherModel, self).__init__()
        # 1 hidden layer of some dimension -- probably need to change ? 
        self.layers =nn.Sequential(nn.Linear(input_dim, hidden_layers ), 
                              nn.Tanh(), 
                              nn.Linear(hidden_layers,hidden_layers +1  ), 
                              nn.Tanh())
        self.actor = nn.Linear(hidden_layers +1, output_dim)
        self.critic = nn.Linear(hidden_layers +1, 1 )

        # Initialize training parameters
        self.timesteps_per_batch = 2048                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200           # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.iters = 200000

		# Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
    
    def forward(self, x): 
        x = self.layers(x)
        actor = self.actor(x)
        critic = self.critic(x)
        
        return actor, critic

   

            
    
    def rollout(self, env, device):
        rollouts = Transition([], [], [], [], [])
        with torch.no_grad(): 
            for i in range(self.timesteps_per_batch):
                states = env.reset()[0]
                done = False

                for j in range(self.max_timesteps_per_episode):
                    '''
                    Get action from policy model
                    '''
                    states = torch.from_numpy(np.array(states)).float().to(device)
                    actor, critic = self.forward(states)
                    action_dist = Categorical(logits=actor.unsqueeze(-2))
                    action = action_dist.probs.argmax(-1)
                    prob = action_dist.log_prob(action)
                    action = action.numpy()[0]
                    
                    
                    new_states, rewards, done, _, _ = env.step(action)

                    rollouts.state.append(states)
                    rollouts.action.append(action)
                    rollouts.reward.append(rewards)
                    rollouts.new_state.append(new_states)
                    rollouts.log_prob.append(prob.numpy())

                    states = new_states
                    if done:
                        break

        return rollouts

    def evaluate(self, batch_states, batch_acts):
        actor, _ = self.forward(batch_states)
        
        action_dist = Categorical(logits=actor.unsqueeze(-2))
        action = action_dist.probs.argmax(-1)
        log_prob = action_dist.log_prob(action)

        return log_prob

    
class StudentModel(nn.Module): 
    def __init__(self, input_dim, output_dim): 
        super(StudentModel, self).__init__()
        hidden_dim = 32
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_dim) 
                                    )
        
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 200 
        
    def forward(self, x): 
        out = self.layers(x)
        return out

    def validate(self, env):
        rewards = []
        log_probs = []
        states = env.reset()[0]
        done = False
        
        for i in range(self.timesteps_per_batch): 
            for j in range(self.max_timesteps_per_episode):
                Q = self.forward(states)
                action_dist = Categorical(logits=Q.unsqueeze(-2))
                action = action_dist.probs.argmax(-1)
                log_prob = action_dist.log_prob(action)
                action = action.numpy()
                new_states, reward, dones, infos, _ = env.step(action)
                
                rewards = np.append(rewards, reward)
                log_probs = np.append(log_probs, log_prob)
                states = new_states
                if done:
                    break
                
        return rewards, log_probs

class TeacherReplayBuffer(object):
    def __init__(self, size, state_dim, action_dim):
        self.size = size 
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.memory = Transition([],[],[],[],[])
        
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

class Training():
    def __init__(self): 
        self.timesteps_per_batch = 2048                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 200           # Max number of timesteps per episode
        self.n_updates_per_iteration = 10                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.iters = 200000
        
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations
        self.seed = None                                # Sets the seed of our program, used for reproducibility of results
        self.reset_every = 50
        self.update_every = 50
        self.eta0 = .75
       
       
    def surprise_reward(self,teacher_reward, t_prob, student_reward, student_policy, states): 
        with torch.no_grad(): 
            eta1 = self.eta0/(max(1, abs((1/teacher_reward.shape[0])*torch.sum(teacher_reward))))
            eta2 = self.eta0/(max(1, abs(1/student_reward.shape[0])*torch.sum(student_reward)))
        
           
            #need student log -prob
            Q = student_policy(states)
            #action_dist = Categorical(logits=Q.unsqueeze(-2))
            action_dist = Categorical(logits = Q)
            action = action_dist.probs.argmax(-1)
            s_prob = action_dist.log_prob(action)
            
            surprise_reward = teacher_reward + eta1.item()*t_prob.clone().detach() + eta2*s_prob
        return  surprise_reward
    
    
    def train_loop(self, env, teacher_model, student_model, student_tar): 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        teacher_a_optim = torch.optim.Adam(teacher_model.parameters(), lr = self.lr)
        teacher_c_optim = torch.optim.Adam(teacher_model.parameters(), lr = self.lr)
        student_optim = torch.optim.Adam(student_model.parameters(), lr = self.lr )
        
        batch_size = self.timesteps_per_batch*self.max_timesteps_per_episode
        
        student_rewards = torch.zeros(batch_size)
        
        
        states = env.reset()[0]

        state_dim = env.observation_space.shape[0]
        action_dim= env.action_space.n
        teacher_memory = TeacherReplayBuffer(100, state_dim, action_dim)

        reset_count = 0
        update_count = 0 
        teacher_model.train()
        student_model.train()
        for i in range(self.iters):
            print("Starting Iteration number: ", i)
            # if (i%self.render_every_i == 0):
            #     env.render()
            #collect teacher rollouts 
            rollouts = teacher_model.rollout(env, device)
            print("rollout ended")
            teacher_memory.insert(rollouts)
            cl_func = torch.nn.MSELoss()
        
            states = torch.vstack(rollouts.state)
            new_states = torch.from_numpy(np.asarray(rollouts.new_state)).float()
            actions = torch.from_numpy(np.asarray(rollouts.action))
            rewards = torch.from_numpy(np.asarray(rollouts.reward)).float()
            probs =  torch.from_numpy(np.asarray(rollouts.log_prob))

            actor, value = teacher_model.forward(states)
            
            rew_sup = self.surprise_reward(rewards, probs, student_rewards, student_model, states)        
            A = rew_sup - value.detach()
            #A = rewards - value.detach()
            #normalize advantage ? 
            A = (A - A.mean()) / (A.std()+1e-10) 
            
            # critic_loss = nn.MSELoss()

            for j in range(self.n_updates_per_iteration):             
                
                actor, value = teacher_model.forward(states)
                action_dist = Categorical(logits=actor.unsqueeze(-2))
                action = action_dist.probs.argmax(-1)
                curr_prob = action_dist.log_prob(action)
                
                
                ratio = torch.exp(curr_prob - probs)
                
                sur1 = ratio*A
                sur2 = torch.clamp(ratio, .8, 1.2)*A 
                
                al = (-torch.min(sur1, sur2)).mean()
                rewards = torch.reshape(rewards, (-1,))
                cl = nn.MSELoss()(value, rewards)
                
                teacher_a_optim.zero_grad()
                al.backward(retain_graph = True)
                teacher_a_optim.step()
                
                teacher_c_optim.zero_grad()
                cl.backward()
                teacher_c_optim.step
                
            
            for k in range(self.n_updates_per_iteration): 
                student_loss = torch.nn.HuberLoss()
                
                batch = teacher_memory.sample_batch(batch_size)
                batch_states = torch.vstack(batch.state)
                batch_new_states = torch.from_numpy(np.asarray(batch.new_state)).float()
                batch_actions = torch.from_numpy(np.asarray(batch.action))
                batch_rewards = torch.from_numpy(np.asarray(batch.reward)).float()
                batch_probs =  torch.from_numpy(np.asarray(batch.log_prob))
                
                q_vals = student_model(batch_states)
                action_q = torch.gather(q_vals, 1, batch_actions)
                
                tar_q_vals = student_tar(batch_new_states)
                tar_action_q = torch.max(tar_q_vals, 1)
                q = rewards + self.gamma*tar_action_q
                
                student_optim.zero_grad()
                loss = student_loss(action_q, q)
                loss.backward()
                student_optim.step()
                
            #test student network -- rollouts? 
            #update student reward 
            student_rewards, _ = student_model.student_validate(env)
            #update target network every %% 
            update_count += 1 
            if update_count == self.update_every: 
                for target_param, param in zip(student_tar.parameters(), student_model.parameters()):
                    target_param.data.copy_(param.data)
                update_count = 0 
            
            reset_count += 1 
            if reset_count == self.reset_every: 
                states = env.reset()
        
