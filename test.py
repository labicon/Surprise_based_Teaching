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
from surprise.policy.DiscretePolicy import DiscreteMLPPolicy

from StudentTeacherAlgo_Diff import Curriculum_Diff as Curriculum 
from trainer_diff import Trainer


########################################################################################
#SparseHopper Env.
@wrap_experiment(log_dir = './experiments/Sparse_HalfCheetah_Diffangle/MaxTRPO', snapshot_mode='all', archive_launch_repo=False)
def Max_TRPO_SparseHalfCheetah(ctxt=None, seed=1):
    """Trust Region Policy Optimization 
    Train a single agent maximizing its own 
    Sparse Hopper environment
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    teacher_env = GymEnv('SparseHalfCheetahAngle-v2')
    student_env = GymEnv('SparseHalfCheetahAngleLimited-v2')

    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(teacher_env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    student_policy = GaussianMLPPolicy(student_env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    value_function = GaussianMLPValueFunction(env_spec=teacher_env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = student_env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500) 
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "replay": None}
    
    teacher_sampler = MaxCustomSampler(envs = teacher_env,  
                           agents = teacher_policy, 
                           worker_factory = MaxSurpriseWorkerFactory, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    algo = Curriculum(teacher_env_spec = teacher_env.spec,
                      student_env_spec = student_env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 5000)
    
    
    trainer.setup( algo = algo, env = teacher_env, student_env=student_env)
    trainer.train(n_epochs = 1000, batch_size = 5000, store_episodes=True)

@wrap_experiment(log_dir = './experiments/Sparse_HalfCheetah_Diffangle/curriculum', snapshot_mode='all', archive_launch_repo=False)
def curriculum_student_teacher_SparseHalfCheetha(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    Sparse_hopper environment
    """
    set_seed(seed)
    teacher_env = GymEnv('SparseHalfCheetahAngle-v2')
    student_env = GymEnv('SparseHalfCheetahAngleLimited-v2')

    trainer = Trainer(ctxt)
    
    teacher_policy = GaussianMLPPolicy(teacher_env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    student_policy = GaussianMLPPolicy(student_env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    
    
    value_function = GaussianMLPValueFunction(env_spec=teacher_env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    student_sampler = CustomSampler(envs = student_env,  
                           agents = student_policy, 
                           worker_factory = SurpriseWorkerFactory,  
                           max_episode_length = 500)
    
    surprise = {"surprise": True, "student": student_policy, "eta0": 0.001, "student_eta0": 0.01, "replay": student_sampler}
    
    teacher_sampler = CustomSampler(envs = teacher_env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = 500)
    
    
    
    algo = Curriculum(teacher_env_spec = teacher_env.spec,
                      student_env_spec = student_env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = 5000)
    
    trainer.setup( algo = algo, env = teacher_env, student_env=student_env)
    trainer.train(n_epochs = 1000, batch_size = 5000, store_episodes=True)


#curriculum_student_teacher_SparseHalfCheetha(seed = 1)
Max_TRPO_SparseHalfCheetah(seed = 1)
#curriculum_student_teacher_SparseHalfCheetha(seed = 2)
#Max_TRPO_SparseHalfCheetah(seed = 2)
#curriculum_student_teacher_SparseHalfCheetha(seed = 3)
#Max_TRPO_SparseHalfCheetah(seed = 3)


#run to test & visualize policy 
# from garage.experiment import Snapshotter

# snapshotter = Snapshotter()
# data = snapshotter.load('./experiments/Sparse_HalfCheetah/curriculum_1')
# policy = data['algo'].policy
# # You can also access other components of the experiment
# env = data['env']
# # from load_rollout import rollout
# from garage import rollout
# path = rollout(env, policy, animated=True)

# np.save('load_rollout.npy', path)