#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 19:34:10 2023

@author: w044elc
"""

import torch

from garage.torch.algos import VPG
from garage.torch.optimizers import (ConjugateGradientOptimizer,
                                     OptimizerWrapper)
import collections
import copy
from garage import EpisodeBatch, StepType
from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage.np import discount_cumsum
from garage.np.algos import RLAlgorithm
from garage.torch import compute_advantages, filter_valids
from garage.torch.optimizers import OptimizerWrapper


def surp_log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    ext_reward = []
    for eps in batch.split():
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        ext_reward.append(eps.extrinsic_rewards)
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('Average Extrinsic Reward', np.mean(ext_reward))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns
class Teacher(VPG):
    """Trust Region Policy Optimization (TRPO).

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 value_function,
                 sampler,
                 policy_optimizer=None,
                 vf_optimizer=None,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy'):

        if policy_optimizer is None:
            policy_optimizer = OptimizerWrapper(
                (ConjugateGradientOptimizer, dict(max_constraint_value=0.01)),
                policy)
        if vf_optimizer is None:
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=2.5e-4)),
                value_function,
                max_optimization_epochs=10,
                minibatch_size=64)
            
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         sampler=sampler,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         num_train_per_epoch=num_train_per_epoch,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)
        #self.sampler = sampler
    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        with torch.no_grad():
            old_ll = self._old_policy(obs)[0].log_prob(actions)

        new_ll = self.policy(obs)[0].log_prob(actions)
        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        return surrogate

    def _train_policy(self, obs, actions, rewards, advantages):
        r"""Train the policy.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """
        self._policy_optimizer.zero_grad()
        loss = self._compute_loss_with_adv(obs, actions, rewards, advantages)
        loss.backward()
        self._policy_optimizer.step(
            f_loss=lambda: self._compute_loss_with_adv(obs, actions, rewards,
                                                       advantages),
            f_constraint=lambda: self._compute_kl_constraint(obs))

        return loss

    
    
    
    """Implementation of Behavioral Cloning in PyTorch."""
# yapf: disable
import itertools

from dowel import tabular
import numpy as np
import torch

from garage import (_Default, log_performance, make_optimizer,
                    obtain_evaluation_episodes, TimeStepBatch)
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.np.policies import Policy
from garage.sampler import Sampler
from garage.torch import as_torch
from garage.experiment import Snapshotter

class Student(RLAlgorithm):
    """Behavioral Cloning.

    Based on Model-Free Imitation Learning with Policy Optimization:
        https://arxiv.org/abs/1605.08478

    Args:
        env_spec (EnvSpec): Specification of environment.
        learner (garage.torch.Policy): Policy to train.
        batch_size (int): Size of optimization batch.
        source (Policy or Generator[TimeStepBatch]): Expert to clone. If a
            policy is passed, will set `.policy` to source and use the trainer
            to sample from the policy.
        sampler (garage.sampler.Sampler): Sampler. If source is a policy, a
            sampler is required for sampling.
        policy_optimizer (torch.optim.Optimizer): Optimizer to be used to
            optimize the policy.
        policy_lr (float): Learning rate of the policy optimizer.
        loss (str): Which loss function to use. Must be either 'log_prob' or
            'mse'. If set to 'log_prob' (the default), `learner` must be a
            `garage.torch.StochasticPolicy`.
        minibatches_per_epoch (int): Number of minibatches per epoch.
        name (str): Name to use for logging.

    Raises:
        ValueError: If learner` is not a
            `garage.torch.StochasticPolicy` and loss is 'log_prob'.

    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        env_spec,
        learner,
        *,
        batch_size,
        source=None,
        sampler=None,
        policy_optimizer=torch.optim.Adam,
        policy_lr=_Default(1e-3),
        loss='log_prob',
        minibatches_per_epoch=16,
        name='BC',
    ):
        self._source = source
        self.learner = learner
        self._optimizer = make_optimizer(policy_optimizer,
                                         module=self.learner,
                                         lr=policy_lr)
        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._minibatches_per_epoch = minibatches_per_epoch
        self._eval_env = None
        self._batch_size = batch_size
        self._name = name

        # For plotting
        self.policy = self.learner

        # Public fields for sampling.
        self._env_spec = env_spec
        self.exploration_policy = None
        self.policy = None
        self.max_episode_length = env_spec.max_episode_length
        self._sampler = sampler
        if isinstance(self._source, Policy):
            self.exploration_policy = self._source
            self._source = source
            if not isinstance(self._sampler, Sampler):
                raise TypeError('Source is a policy. Missing a sampler.')
        else:
            self._source = itertools.cycle(iter(source))

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, for services such as
                snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for epoch in trainer.step_epochs():
            if self._eval_env is not None:
                log_performance(epoch,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env),
                                discount=1.0)
            losses = self._train_once(trainer, epoch)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))

    def _train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        batch = self._obtain_samples(trainer, epoch)
        indices = np.random.permutation(len(batch.actions))
        minibatches = np.array_split(indices, self._minibatches_per_epoch)
        losses = []
        for minibatch in minibatches:
            observations = as_torch(batch.observations[minibatch])
            actions = as_torch(batch.actions[minibatch])
            self._optimizer.zero_grad()
            loss = self._compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        return losses

    def _obtain_samples(self, trainer, epoch):
        """Obtain samples from self._source.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            TimeStepBatch: Batch of samples.

        """
        if isinstance(self._source, Policy):
            batch = trainer.obtain_episodes(epoch)
            log_performance(epoch, batch, 1.0, prefix='Expert')
            return batch
        else:
            batches = []
            while (sum(len(batch.actions)
                       for batch in batches) < self._batch_size):
                batches.append(next(self._source))
            return TimeStepBatch.concatenate(*batches)

    def _compute_loss(self, observations, expert_actions):
        """Compute loss of self._learner on the expert_actions.

        Args:
            observations (torch.Tensor): Observations used to select actions.
                Has shape :math:`(B, O^*)`, where :math:`B` is the batch
                dimension and :math:`O^*` are the observation dimensions.
            expert_actions (torch.Tensor): The actions of the expert.
                Has shape :math:`(B, A^*)`, where :math:`B` is the batch
                dimension and :math:`A^*` are the action dimensions.

        Returns:
            torch.Tensor: The loss through which gradient can be propagated
                back to the learner. Depends on self._loss.

        """
        learner_output = self.learner(observations)
        if self._loss == 'mse':
            if isinstance(learner_output, torch.Tensor):
                # We must have a deterministic policy as the learner.
                learner_actions = learner_output
            else:
                # We must have a StochasticPolicy as the learner.
                action_dist, _ = learner_output
                learner_actions = action_dist.rsample()
            return torch.mean((expert_actions - learner_actions)**2)
        else:
            assert self._loss == 'log_prob'
            # We already checked that we have a StochasticPolicy as the learner
            action_dist, _ = learner_output
            return -torch.mean(action_dist.log_prob(expert_actions))
        
class Curriculum(VPG): 
    def __init__(self, env_spec,
                 teacher_policy,
                 student_policy,
                 teacher_sampler,
                 teacher_value_function,
                 student_sampler,
                 batch_size,
                 student_policy_optimizer=torch.optim.Adam,
                 policy_lr=_Default(1e-3),
                 loss='log_prob',
                 minibatches_per_epoch=16,
                 name='BC'):
        '''
        teacher_policy_optimizer=None,
        vf_optimizer=None,
        num_train_per_epoch=1,
        discount=0.99,
        gae_lambda=0.98,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy'):
        '''
        
       
        self.teacher_sampler = teacher_sampler
        self.teacher_policy = teacher_policy
        self.teacher = Teacher(env_spec = env_spec, 
                               policy = self.teacher_policy, 
                               value_function = teacher_value_function, 
                               sampler = self.teacher_sampler)
        self.policy = teacher_policy
        self._sampler = teacher_sampler
        
        
        #intializing student algorithm parameters
        self._source = self.teacher_policy
        self.learner = student_policy
        self._student_optimizer = make_optimizer(student_policy_optimizer,
                                         module=self.learner,
                                         lr=policy_lr)
        
        self.student_policy = student_policy 
        self.student_sampler = student_sampler
        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._minibatches_per_epoch = minibatches_per_epoch
        self._eval_env = None
        self._batch_size = batch_size
        self._name = name

        # For plotting
        #self.policy = self.learner

        # Public fields for sampling.
        self._env_spec = env_spec
        self.exploration_policy = None
        #self.policy = None
        self.max_episode_length = env_spec.max_episode_length
        #self._sampler =
        if isinstance(self._source, Policy):
            self.exploration_policy = self._source
            self._source = self.teacher_policy
            if not isinstance(self._sampler, Sampler):
                raise TypeError('Source is a policy. Missing a sampler.')
        else:
            self._source = itertools.cycle(iter(self.teacher_policy))
    
    def train(self, trainer):
        #self.teacher.train(trainer)
        last_return = None
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for _ in trainer.step_epochs():
            #trainer.policy = self.teacher_policy
            self.policy = self.teacher_policy
            #trainer._sampler = self.teacher_sampler
            self._sampler = self.teacher_sampler
            for _ in range(self.teacher._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr)
                last_return = self.teacher._train_once(trainer.step_itr, eps)
                trainer.step_itr += 1
            
            if self._eval_env is not None:
                log_performance(_,
                                obtain_evaluation_episodes(
                                    self.teacher_policy, self._eval_env),
                                discount=1.0, prefix = 'TeacherEval')
                
            #self.teacher_policy = trainer.policy 
            self._source = self.teacher_policy 
            #trainer.policy = self.learner 
            self.policy = self.learner
            #trainer._sampler = self.student_sampler
            self._sampler = self.student_sampler
            
            if self._eval_env is not None:
                log_performance(_,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env),
                                discount=1.0)
            losses = self.student_train_once(trainer, _)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
    
                tabular.record('StdLoss', np.std(losses))
            #self.learner = trainer.policy 
            self.teacher._sampler.student = self.learner
            
        return last_return
    
        
    
    def student_train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, for services such as
                snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for epoch in trainer.step_epochs():
            if self._eval_env is not None:
                log_performance(epoch,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env),
                                discount=1.0)
            losses = self.student_train_once(trainer, epoch)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))

    def student_train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        batch = self.student_obtain_samples(trainer, epoch)
        indices = np.random.permutation(len(batch.actions))
        minibatches = np.array_split(indices, self._minibatches_per_epoch)
        losses = []
        for minibatch in minibatches:
            observations = as_torch(batch.observations[minibatch])
            actions = as_torch(batch.actions[minibatch])
            self._student_optimizer.zero_grad()
            loss = self.student_compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._student_optimizer.step()
        return losses

    def student_obtain_samples(self, trainer, epoch):
        """Obtain samples from self._source.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            TimeStepBatch: Batch of samples.

        """
        if isinstance(self._source, Policy):
            batch = trainer.obtain_episodes(epoch)
            log_performance(epoch, batch, 1.0, prefix='Expert')
            return batch
        else:
            batches = []
            while (sum(len(batch.actions)
                       for batch in batches) < self._batch_size):
                batches.append(next(self._source))
            return TimeStepBatch.concatenate(*batches)

    def student_compute_loss(self, observations, expert_actions):
        """Compute loss of self._learner on the expert_actions.

        Args:
            observations (torch.Tensor): Observations used to select actions.
                Has shape :math:`(B, O^*)`, where :math:`B` is the batch
                dimension and :math:`O^*` are the observation dimensions.
            expert_actions (torch.Tensor): The actions of the expert.
                Has shape :math:`(B, A^*)`, where :math:`B` is the batch
                dimension and :math:`A^*` are the action dimensions.

        Returns:
            torch.Tensor: The loss through which gradient can be propagated
                back to the learner. Depends on self._loss.

        """
        learner_output = self.learner(observations)
        if self._loss == 'mse':
            if isinstance(learner_output, torch.Tensor):
                # We must have a deterministic policy as the learner.
                learner_actions = learner_output
            else:
                # We must have a StochasticPolicy as the learner.
                action_dist, _ = learner_output
                learner_actions = action_dist.rsample()
            return torch.mean((expert_actions - learner_actions)**2)
        else:
            assert self._loss == 'log_prob'
            # We already checked that we have a StochasticPolicy as the learner
            action_dist, _ = learner_output
            return -torch.mean(action_dist.log_prob(expert_actions))
