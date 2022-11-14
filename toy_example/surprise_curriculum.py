# -*- coding: utf-8 -*-
import gym 
import torch 
import torch.nn as nn
import numpy as np 
import math 
from collections import deque, namedtuple
from torch.distributions.categorical import Categorical 
import cl_policies
'''
initialize environment 
'''
env = gym.make('MountainCar-v0')
observation, infor = env.reset()

'''       
def suprisal_reward(teacher_reward, t_prob, student_reward, student_policy, eta0, states, actions, device): 
    #n = max(1, abs((1/teacher_reward.size)*np.sum(teacher_reward)))
    with torch.no_grad(): 
        eta1 = eta0/(max(1, abs((1/teacher_reward.shape[0])*torch.sum(teacher_reward))))
        eta2 = eta0/(max(1, abs(1/student_reward.shape[0])*torch.sum(student_reward)))
    
       
        #need student log -prob
        Q = student_policy(states)
        #action_dist = Categorical(logits=Q.unsqueeze(-2))
        action_dist = Categorical(logits = Q)
        action = action_dist.probs.argmax(-1)
        s_prob = action_dist.log_prob(action)
        
        surprise_reward = teacher_reward + eta1.item()*t_prob.clone().detach() + eta2*s_prob
    return  surprise_reward
 '''
def evaluate_teacher(teacher, env):
    episodes = 50

    performance_list = []
    teacher.test()
    for epi in range(episodes):
        s = env.reset()

        done = False
        cum_reward = 0.0
        while not done:
            actor, value = teacher(s)
            ns, reward, done, _ = env.step(actor)
            s = ns
            cum_reward += reward
        
        performance_list.append(cum_reward)
    
    return np.sum(performance_list)

def evaluate_student(student, env): 
    episodes = 50 
    performance_list = []
    student.test()
    for epi in range(episodes):
        s = env.reset()

        done = False
        cum_reward = 0.0
        while not done:
            Q  = student(s)
            act = np.armax(Q)
            ns, reward, done, _ = env.step(act)
            s = ns
            cum_reward += reward
        
        performance_list.append(cum_reward)
    
    return np.sum(performance_list)

if __name__ == "__main__":
    action_space = env.action_space.n
    state_space = env.observation_space.shape[0]

    teacher = cl_policies.TeacherModel(state_space, action_space, 32)
    student = cl_policies.StudentModel(state_space, action_space)

    print("Training start")
    train = cl_policies.Training()
    train.train_loop(env, teacher, student, student)
    print("Training ended, Evaluation start")
    t_perf = evaluate_teacher(teacher, env)
    s_perf = evaluate_student(student, env)
    
    print("Teacher rewards: " + t_perf)
    print("student rewards: "+ s_perf)
