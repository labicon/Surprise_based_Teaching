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
from garage.replay_buffer import ReplayBuffer
from setuptools import setup 
from StudentTeacherAlgo import Curriculum 
from DiscretePolicy import DiscreteMLPPolicy


########################################################################################
#Reacher Env.
@wrap_experiment(log_dir = './experiments/Reacher/MaxTRPO')
def Max_TRPO_Reacher(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    Reacher environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('Reacher-v2')
    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
 
    
    surprise = {"surprise": False, "student": None, "eta0": 0.01, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 50)
    
    algo = TRPO(env_spec = env.spec, 
               policy=teacher_policy,
               value_function=value_function,
               discount=0.995,
               gae_lambda= .95,
               center_adv=False, 
               sampler = teacher_sampler)
    
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/Reacher/curriculum')
def curriculum_student_teacher_Reacher(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Reacher environment
    """
    set_seed(seed)
    env = GymEnv('Reacher-v2')
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
                           max_episode_length = 50)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.01, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 50)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

# Max_TRPO_Reacher(seed = 1)
# curriculum_student_teacher_Reacher(seed = 1)

########################################################################################
#Swimmer Env.
@wrap_experiment(log_dir = './experiments/Swimmer/MaxTRPO')
def Max_TRPO_Swimmer(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    Swimmer environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('Swimmer-v2')
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
                           max_episode_length = 1000) 
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.01, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 1000)
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/Swimmer/curriculum')
def curriculum_student_teacher_Swimmer(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Swimmer environment
    """
    set_seed(seed)
    env = GymEnv('Swimmer-v2')
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
                           max_episode_length = 1000)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 1000)
    
    
    
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
#SparseHalfCheetah Env.
@wrap_experiment(log_dir = './experiments/Sparse_HalfCheetah/MaxTRPO')
def Max_TRPO_SparseHalfCheetah(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    Sparse HalfCheetah environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('SparseHalfCheetah-v2')
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
                           max_episode_length = 1000) 
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 1000)
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 1000, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/Sparse_HalfCheetah/curriculum')
def curriculum_student_teacher_SparseHalfCheetah(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Sparse_halfcheetah environment
    """
    set_seed(seed)
    env = GymEnv('SparseHalfCheetah-v2')
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
                           max_episode_length = 1000)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 1000)
    
    
    
    algo = Curriculum(env_spec = env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 1000)
    
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 1000, batch_size = 1000)

Max_TRPO_SparseHalfCheetah(seed = 1)
curriculum_student_teacher_SparseHalfCheetah(seed = 1)
Max_TRPO_SparseHalfCheetah(seed = 2)
curriculum_student_teacher_SparseHalfCheetah(seed = 2)
Max_TRPO_SparseHalfCheetah(seed = 3)
curriculum_student_teacher_SparseHalfCheetah(seed = 3)
Max_TRPO_SparseHalfCheetah(seed = 4)
curriculum_student_teacher_SparseHalfCheetah(seed = 4)
Max_TRPO_SparseHalfCheetah(seed = 5)
curriculum_student_teacher_SparseHalfCheetah(seed = 5)

'''
#run to test & visualize policy 
from garage.experiment import Snapshotter
snapshotter = Snapshotter()
data = snapshotter.load('./experiments/Reacher/MaxTRPO_7')
policy = data['algo'].policy
# You can also access other components of the experiment
env = data['env']
from garage import rollout
path = rollout(env, policy, animated=True)
'''
