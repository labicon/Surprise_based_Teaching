#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:55:19 2022
@author: w044elc
"""
from garage.sampler import Sampler, LocalSampler
from garage.torch.algos import PPO, TRPO, BC
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.q_functions import DiscreteMLPQFunction
from garage.trainer import Trainer
import torch
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
import numpy as np 
from garage import EpisodeBatch, StepType
from garage.experiment import deterministic
from garage.sampler import _apply_env_update
from garage.sampler.worker import Worker
import psutil
from garage.experiment.deterministic import get_seed
import abc 
import copy 
from collections import defaultdict
from garage import EpisodeBatch
import SurpriseFunctions
from SurpriseFunctions import SurpriseWorkerFactory, CustomSampler, SurpriseWorker
from MaxSurpriseFunctions import MaxSurpriseWorkerFactory, MaxCustomSampler, MaxSurpriseWorker
from StudentOnlyMaxSurpriseFunctions import SOMaxSurpriseWorkerFactory, SOCustomSampler, SOMaxSurpriseWorker
from trpo_discrete import DisTRPO
from garage.replay_buffer import ReplayBuffer
from SparseRewardMountainCar import SparseContinuous_MountainCarEnv
from setuptools import setup 
from StudentTeacherAlgo import Curriculum 
from DiscreteStudentTeacherAlgo import Dis_Curriculum 
from DiscretePolicy import DiscreteMLPPolicy


#Mountain Car
@wrap_experiment(log_dir = './experiments/MountainCar/MaxTRPO')
def Max_TRPOMC(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    Sparse reward Mountain Car environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('Sparse_MountainCar-v0')
    #env = GymEnv('CartPoleSwingup-v0')
    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
 
    
    surprise = {"surprise": True, "student": None, "eta0": 0.01, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    

    
    student_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    student_sampler = CustomSampler(envs = env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
  
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)




@wrap_experiment(log_dir = './experiments/MountainCar/Curriculum')
def curriculum_student_teacherMC(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Sparse reward Mountain car environment
    """
    set_seed(seed)
    env = GymEnv('Sparse_MountainCar-v0')
    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    student_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.01, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)
    


########################################################################################
#Cartpole 
@wrap_experiment(log_dir = './experiments/CartPoleSwingUp/MaxTRPO')
def Max_TRPO_CP(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    CartPole environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('CartPoleSwingup-v1')
    trainer = Trainer(ctxt)
    

    
    teacher_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    student_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
 
    
    surprise = {"surprise": True, "student": None, "eta0": 0.001, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 200)
    
    

    student_sampler = CustomSampler(envs = env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 200)
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 500, batch_size = 1000)
    


@wrap_experiment(log_dir = './experiments/CartPoleSwingUp/curriculum')
def curriculum_student_teacher_CP(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    CartPole environment
    """
    set_seed(seed)
    env = GymEnv('CartPoleSwingup-v1')
    trainer = Trainer(ctxt)
    
    teacher_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    student_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 200)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 200)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 500, batch_size = 1000)



################################################################################################################################################
'''
Mountain Car with different constraints: 
    Student has less power available than teacher --> requires more force to move the same distance 
'''
from StudentTeacherAlgo_Diff import Curriculum_Diff
@wrap_experiment(log_dir = './experiments/MountainCar_DiffForce/Curriculum')
def curriculum_student_teacherMC(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Sparse reward Mountain car environment
    """
    set_seed(seed)
    env = GymEnv('Sparse_MountainCar-v0')
    student_env = env 
    student_env.student = True
    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    student_policy = GaussianMLPPolicy(student_env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = student_env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.01, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    
    
    algo = Curriculum_Diff(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)
  
'''
CartPole Swing Up with different constraints: 

'''
from StudentTeacherAlgo_CPDiff import Curriculum 
@wrap_experiment(log_dir = './experiments/CartPoleDiffPos/curriculum_diff')
def curriculum_student_teacher_CPDiff(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    CartPole environment
    """
    set_seed(seed)
    env = GymEnv('CartPole-v1')
    '''
    V1 cartpole swingup differing constraints are in x-position bounds
        student allowed x-position bound is +- (1/2) teacher x-position 
    '''
    student_env = env 
    student_env.student = True
    trainer = Trainer(ctxt)
    
    teacher_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    student_policy = DiscreteMLPPolicy(student_env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = student_env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.005, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)
    

@wrap_experiment(log_dir = './experiments/CartPoleDiffMass/curriculum')
def curriculum_student_teacher_CPDiff(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    CartPole environment
    """
    set_seed(seed)
    env = GymEnv('CartPole-v2')
    '''
    V2 cartpole swingup differing constraints are in pole mass 
        student pole mass = .12, teacher pole mass = .1
    '''
    student_env = env 
    student_env.student = True
    trainer = Trainer(ctxt)
    
    teacher_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    student_policy = DiscreteMLPPolicy(student_env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = student_env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.005, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/CartPole/MaxTRPO_diff')
def Max_TRPO_CPDiff(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    CartPole environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('CartPole-v1')
    trainer = Trainer(ctxt)
    

    
    teacher_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    student_policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
 
    
    surprise = {"surprise": True, "student": None, "eta0": 0.005, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    

    student_sampler = CustomSampler(envs = env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)
    


'''
#run to test & visualize policy 
from garage.experiment import Snapshotter
snapshotter = Snapshotter()
data = snapshotter.load('./experiments/CartPoleSwingUp/curriculum_11')
policy = data['algo'].policy
# You can also access other components of the experiment
env = data['env']
from garage import rollout
path = rollout(env, policy, animated=True)
'''