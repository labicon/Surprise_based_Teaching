#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:04:19 2023
@author: kh-ryu
"""
from garage.sampler import RaySampler
from garage.torch.algos import PPO, TRPO
from garage.torch.policies import GaussianMLPPolicy
from policy.DiscretePolicy import DiscreteMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

import torch
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.trainer import Trainer

@wrap_experiment(log_dir = './experiments/MountainCar/PPO_baseline', archive_launch_repo=False)
def PPO_MountainCar(ctxt=None, seed=1):
    """Train PPO with sparse reward on SparseMountainCar-v0.
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('Sparse_MountainCar-v0')
    trainer = Trainer(ctxt)
    
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)    
    
    sampler = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=500)    
    
    algo = PPO(env_spec = env.spec, 
               policy=policy,
               value_function=value_function,
               discount=0.99,
               sampler = sampler)
    
  
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/MountainCar/TRPO_baseline', archive_launch_repo=False)
def TRPO_MountainCar(ctxt=None, seed=1):
    """Train TRPO with sparse reward on SparseMountainCar-v0.
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('Sparse_MountainCar-v0')
    trainer = Trainer(ctxt)
    
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)    
    
    sampler = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=500)    
    
    algo = TRPO(env_spec = env.spec, 
               policy=policy,
               value_function=value_function,
               discount=0.99,
               sampler = sampler,
               center_adv=False)
    
  
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 200, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/CartpoleSwingup/PPO_baseline', archive_launch_repo=False)
def PPO_CartPoleSwingUP(ctxt=None, seed=1):
    """Train PPO with sparse reward on SCartpoleSwingup-v1.
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('CartPoleSwingUp-v1')
    trainer = Trainer(ctxt)
    
    policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)    
    
    sampler = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=500)    
    
    algo = PPO(env_spec = env.spec, 
               policy=policy,
               value_function=value_function,
               discount=0.95,
               sampler = sampler)
    
  
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 500, batch_size = 1000)

@wrap_experiment(log_dir = './experiments/CartpoleSwingup/TRPO_baseline', archive_launch_repo=False)
def TRPO_CartPoleSwingUp(ctxt=None, seed=1):
    """Train TRPO with sparse reward on CartpoleSwingup-v1.
    
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)

    env = GymEnv('CartPoleSwingUp-v1')
    trainer = Trainer(ctxt)
    
    policy = DiscreteMLPPolicy(env.spec,
                               hidden_sizes=[128, 128],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(128, 128),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)    
    
    sampler = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=500)    
    
    algo = TRPO(env_spec = env.spec, 
               policy=policy,
               value_function=value_function,
               discount=0.95,
               sampler = sampler,
               center_adv=False)
    
  
    trainer.setup( algo = algo, env = env)
    trainer.train(n_epochs = 500, batch_size = 1000)


# PPO_CartPoleSwingUP(seed=1)
# TRPO_CartPoleSwingUp(seed=1)
# PPO_CartPoleSwingUP(seed=2)
# TRPO_CartPoleSwingUp(seed=2)
# PPO_CartPoleSwingUP(seed=3)
# TRPO_CartPoleSwingUp(seed=3)