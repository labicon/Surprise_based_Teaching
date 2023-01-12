#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:55:19 2022

@author: w044elc
"""
from garage.sampler import Sampler, LocalSampler
from garage.torch.algos import PPO, TRPO
from garage.torch.policies import GaussianMLPPolicy, DeterministicMLPPolicy, DiscreteQFArgmaxPolicy
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
from SurpriseFunctions import SurpriseWorkerFactory, CustomSampler
from garage.replay_buffer import ReplayBuffer
from ppo_dis import PPO_Discrete
from SparseRewardMountainCar import Continuous_MountainCarEnv
from setuptools import setup 
from gym.envs.registration import register 
from garage.envs import normalize



@wrap_experiment
def ppo(ctxt=None, seed=1):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    #env = GymEnv('MountainCarContinuous-v0')
    env = GymEnv('Sparse_MountainCar-v0')
    #env = GymEnv('CartPole-v1')
    trainer = Trainer(ctxt)
    #replay_buffer = ReplayBuffer(env_spec = env.spec, size_in_transitions= 10000, time_horizon = 500)
    
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    '''
    q_fun = DiscreteMLPQFunction(env_spec = env.spec, hidden_sizes = (32,32))
    policy = DiscreteQFArgmaxPolicy(qf = q_fun, env_spec = env.spec)
    '''
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    surprise = {"surprise": True, "student": None, "eta0": 0.05, "replay":None}
    sampler = CustomSampler(envs = env,  
                           agents = policy, 
                           worker_factory = SurpriseWorkerFactory, 
                           worker_args = surprise, 
     
                           max_episode_length = 500)
    '''
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               center_adv=False, 
               sampler = sampler)
    '''
    algo = TRPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               sampler = sampler)

    trainer.setup(algo, env)
    trainer.train(n_epochs=500, batch_size=1000)


ppo(seed=1)

'''
#tests trained policy
from garage.experiment import Snapshotter

snapshotter = Snapshotter()
data = snapshotter.load('./data/local/experiment/ppo_1')
policy = data['algo'].policy
# You can also access other components of the experiment
env = data['env']

from garage import rollout
path = rollout(env, policy, animated=True)



'''
