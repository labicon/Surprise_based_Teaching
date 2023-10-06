#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:53:32 2022

@author: w044elc
"""
from garage.sampler import Sampler
import torch
import numpy as np 
from garage import EpisodeBatch, StepType
from garage.experiment import deterministic
from garage.sampler import _apply_env_update
from garage.sampler.worker import Worker
import psutil
from garage.experiment.deterministic import get_seed
import copy 
from collections import defaultdict
from garage.sampler.sampler import Sampler
from garage import EpisodeBatch
from model.Regressor import Regressor


class SurpriseWorker(Worker):
    """Initialize a worker.
    Args:
        seed (int): The seed to use to intialize random number generators.
        max_episode_length (int or float): The maximum length of episodes which
            will be sampled. Can be (floating point) infinity.
        worker_number (int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.
    Attributes:
        agent (Policy or None): The worker's agent.
        env (Environment or None): The worker's environment.
    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number, 
            worker_args
            ):
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)
        self.agent = None
        self.env = None
        self._env_steps = []
        self._observations = []
        self._last_observations = []
        self._agent_infos = defaultdict(list)
        self._lengths = []
        self._prev_obs = None
        self._eps_length = 0
        self._episode_infos = defaultdict(list)
        self.worker_init()
        self.surprisal_bonus = None

        if worker_args != {}: 
            self.surprisal_bonus = worker_args["surprise"]
            self.student = worker_args["student"]
            self.student_sampler = worker_args["replay"]
            self.eta0 = worker_args["eta0"]
            if "student_eta0" in worker_args.keys():
                self.student_eta0 = worker_args["student_eta0"]
            else:
                self.student_eta0 = self.eta0
            self.regressor_hidden_size = worker_args["regressor_hidden_size"]
            self.state_dim = worker_args["state_dim"]
            self.action_dim = worker_args["action_dim"]

            self.regressor = Regressor(self.state_dim + self.action_dim, 
                                       self.state_dim, 
                                       self.regressor_hidden_size,
                                       worker_args["regressor_epoch"],
                                       worker_args["regressor_batch_size"])
            self.student_regressor = Regressor(self.state_dim + self.action_dim, 
                                               self.state_dim, 
                                               self.regressor_hidden_size,
                                               worker_args["regressor_epoch"],
                                               worker_args["regressor_batch_size"])
        else:
            self.regressor = None
            self.student_regressor = None

    def SurpriseBonus(self,teacher_reward, student_reward, new_states, states_actions):
        # Teacher surprise wrt the environment
        teacher_log = self.regressor.log_likelihood(states_actions, new_states)
        eta1 = self.eta0 / np.max([1.0, np.mean(np.abs(teacher_reward))])

        teacher_surprise = -eta1*teacher_log
        teacher_surprise = teacher_surprise.reshape(teacher_surprise.shape[0])

        # Student surprise wrt the teacher
        eta2 = self.student_eta0 / np.max([1.0, np.mean(np.abs(student_reward))])
        # next state sampled from teacher regressor
        teacher_new_states = self.regressor.sample(states_actions)
        teacher_log = self.regressor.log_likelihood(states_actions, teacher_new_states)
        student_log = self.student_regressor.log_likelihood(states_actions, teacher_new_states)
        student_surprise = eta2*(teacher_log - student_log)
        student_surprise = student_surprise.reshape(student_surprise.shape[0])

        surprise_reward = teacher_surprise - student_surprise 
        
        new_reward = torch.tensor(teacher_reward) + surprise_reward
        return new_reward

    def calculate_surprise(self, teacher_reward, student_reward, new_states, states_actions):
        # Teacher surprise wrt the environment
        teacher_log = self.regressor.log_likelihood(states_actions, new_states)
        eta1 = self.eta0 / np.max([1.0, np.mean(np.abs(teacher_reward))])

        teacher_surprise = -eta1*teacher_log
        teacher_surprise = teacher_surprise.reshape(teacher_surprise.shape[0])

        # Student surprise wrt the teacher
        eta2 = self.student_eta0 / np.max([1.0, np.mean(np.abs(student_reward))])
        # next state sampled from teacher regressor
        teacher_new_states = self.regressor.sample(states_actions)
        teacher_log = self.regressor.log_likelihood(states_actions, teacher_new_states)
        student_log = self.student_regressor.log_likelihood(states_actions, teacher_new_states)
        student_surprise = eta2*(teacher_log - student_log)
        student_surprise = student_surprise.reshape(student_surprise.shape[0])

        return teacher_surprise, student_surprise   
    
    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            deterministic.set_seed(self._seed + self._worker_number)

    def update_agent(self, agent_update):
        """Update an agent, assuming it implements :class:`~Policy`.
        Args:
            agent_update (np.ndarray or dict or Policy): If a tuple, dict, or
                np.ndarray, these should be parameters to agent, which should
                have been generated by calling `Policy.get_param_values`.
                Alternatively, a policy itself. Note that other implementations
                of `Worker` may take different types for this parameter.
        """
        if isinstance(agent_update, (dict, tuple, np.ndarray)):
            self.agent.set_param_values(agent_update)
        elif agent_update is not None:
            self.agent = agent_update

    def update_env(self, env_update):
        """Use any non-None env_update as a new environment.
        A simple env update function. If env_update is not None, it should be
        the complete new environment.
        This allows changing environments by passing the new environment as
        `env_update` into `obtain_samples`.
        Args:
            env_update(Environment or EnvUpdate or None): The environment to
                replace the existing env with. Note that other implementations
                of `Worker` may take different types for this parameter.
        Raises:
            TypeError: If env_update is not one of the documented types.
        """
        self.env, _ = _apply_env_update(self.env, env_update)

    def start_episode(self):
        """Begin a new episode."""
        self._eps_length = 0
        self._prev_obs, episode_info = self.env.reset()
        for k, v in episode_info.items():
            self._episode_infos[k].append(v)
            
        self.first_state = self._prev_obs
        self.agent.reset()
        return self.env._step_cnt
    
    def step_episode(self):
        """Take a single time-step in the current episode.
        Returns:
            bool: True iff the episode is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.
        """
        if self._eps_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            
            self._observations.append(self._prev_obs)
            
            self._env_steps.append(es)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            self._eps_length += 1

            if not es.terminal:
                self._prev_obs = es.observation
                return False
        self._lengths.append(self._eps_length)
        self._last_observations.append(self._prev_obs)
        return True

    def collect_episode(self):
        """Collect the current episode, clearing the internal buffer.
        Returns:
            EpisodeBatch: A batch of the episodes completed since the last call
                to collect_episode().
        """
        observations = self._observations
        #new_states = torch.tensor(observations)
        self._observations = []
        last_observations = self._last_observations
        #states = torch.tensor(last_observations)
        self._last_observations = []

        self.actions = []
        self.rewards = []
        self.states = []
        self.states.append(self.first_state)
        env_infos = defaultdict(list)
        step_types = []
        ext_rewards = []
        for es in self._env_steps:
            ext_rewards.append(es.reward)
            self.rewards.append(es.reward)
            self.actions.append(es.action)
            self.states.append(es.observation)
            step_types.append(es.step_type)
            for k, v in es.env_info.items():
                env_infos[k].append(v)
            
        if self.surprisal_bonus == True:
            self.states = torch.tensor(self.states)       
            self.new_states = self.states[1:,:]
            self.states = self.states[:-1, :]
            
            actions = torch.tensor(self.actions)
            if actions.dim() == 1: 
            
                actions = actions.unsqueeze(1)
            
            self.state_action = torch.hstack([self.states, actions])
            
            sample = self.student_sampler.obtain_samples(itr = 1, num_samples=500, agent_update = self.student )
            st_obs = sample.observations
            st_n_obs = sample.next_observations
            st_act = sample.actions
            st_rew = sample.rewards
          
            student_new_state = torch.tensor(st_n_obs)
            student_state = torch.tensor(st_obs)
            student_action = torch.tensor(st_act)
            actions = torch.tensor(self.actions)
            if student_action.dim() == 1: 
                student_action = student_action.unsqueeze(1)
            
            student_state_action = torch.hstack([student_state, student_action])
            student_reward = st_rew.reshape(st_rew.shape[0])
            if self.student_regressor == None:
                self.student_regressor = Regressor(student_state_action.shape[1], student_new_state.shape[1], self.regressor_hidden_size)
            self.student_regressor.fit(student_state_action, student_new_state)             
            
            if self.regressor == None:
                self.regressor = Regressor(self.state_action.shape[1], self.new_states.shape[1], self.regressor_hidden_size)
            self.regressor.fit(self.state_action, self.new_states)
            self.rewards = self.SurpriseBonus(self.rewards, student_reward, self.new_states, self.state_action)
            
        self._env_steps = []

        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)

        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)

        episode_infos = self._episode_infos
        self._episode_infos = defaultdict(list)
        for k, v in episode_infos.items():
            episode_infos[k] = np.asarray(v)

        lengths = self._lengths
        self._lengths = []
        return EpisodeBatch(env_spec=self.env.spec,
                            episode_infos=episode_infos,
                            observations=np.asarray(observations),
                            last_observations=np.asarray(last_observations),
                            actions=np.asarray(self.actions),
                            rewards=np.asarray(self.rewards),
                            step_types=np.asarray(step_types, dtype=StepType),
                            env_infos=dict(env_infos),
                            agent_infos=dict(agent_infos),
                            lengths=np.asarray(lengths, dtype='i'))

    def rollout(self):
        """Sample a single episode of the agent in the environment.
        Returns:
            EpisodeBatch: The collected episode.
        """
        self.start_episode()
        while not self.step_episode():
            pass
        episode = self.collect_episode()
        
        return episode

    def shutdown(self):
        """Close the worker's environment."""
        self.env.close()


def identity_function(value):
    """Do nothing.
    This function exists so it can be pickled.
    Args:
        value(object): A value.
    Returns:
        object: The value.
    """
    return value

class SurpriseWorkerFactory:
    """Constructs workers for Samplers.
    The intent is that this object should be sufficient to avoid subclassing
    the sampler. Instead of subclassing the sampler for e.g. a specific
    backend, implement a specialized WorkerFactory (or specify appropriate
    functions to this one). Not that this object must be picklable, since it
    may be passed to workers. However, its fields individually need not be.
    All arguments to this type must be passed by keyword.
    Args:
        max_episode_length(int): The maximum length episodes which will be
            sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.
    """

    def __init__(
            self,
            *,  # Require passing by keyword.
            max_episode_length,
            is_tf_worker=False,
            seed=get_seed(),
            n_workers=psutil.cpu_count(logical=False),
            worker_class=SurpriseWorker,
            worker_args=None):
        self.n_workers = n_workers
        self._seed = seed
        self._max_episode_length = max_episode_length
        if is_tf_worker:
            # Import here to avoid hard dependency on TF.
            # pylint: disable=import-outside-toplevel
            from garage.tf.samplers import TFWorkerClassWrapper
            worker_class = TFWorkerClassWrapper(worker_class)
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args

    def prepare_worker_messages(self, objs, preprocess=identity_function):
        """Take an argument and canonicalize it into a list for all workers.
        This helper function is used to handle arguments in the sampler API
        which may (optionally) be lists. Specifically, these are agent, env,
        agent_update, and env_update. Checks that the number of parameters is
        correct.
        Args:
            objs(object or list): Must be either a single object or a list
                of length n_workers.
            preprocess(function): Function to call on each single object before
                creating the list.
        Raises:
            ValueError: If a list is passed of a length other than `n_workers`.
        Returns:
            List[object]: A list of length self.n_workers.
        """
        if isinstance(objs, list):
            if len(objs) != self.n_workers:
                raise ValueError(
                    'Length of list doesn\'t match number of workers')
            return [preprocess(obj) for obj in objs]
        else:
            return [preprocess(objs) for _ in range(self.n_workers)]

    def __call__(self, worker_number):
        """Construct a worker given its number.
        Args:
            worker_number(int): The worker number. Should be at least 0 and
                less than or equal to `n_workers`.
        Raises:
            ValueError: If the worker number is greater than `n_workers`.
        Returns:
            garage.sampler.Worker: The constructed worker.
        """
        if worker_number >= self.n_workers:
            raise ValueError('Worker number is too big')
        return self._worker_class(worker_number=worker_number,
                                  seed=self._seed,
                                  max_episode_length=self._max_episode_length,
                                  worker_args = self._worker_args)
    
    
    
    
class CustomSampler(Sampler):
    """Sampler that runs workers in the main process.
    This is probably the simplest possible sampler. It's called the "Local"
    sampler because it runs everything in the same process and thread as where
    it was called from.
    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.
    Args:
        agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
            If a list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the
            workers.
        envs (Environment or List[Environment]): Environment from which
            episodes are sampled. If a list is passed in, it must have length
            exactly `worker_factory.n_workers`, and will be spread across the
            workers.
        worker_factory (WorkerFactory): Pickleable factory for creating
            workers. Should be transmitted to other processes / nodes where
            work needs to be done, then workers should be constructed
            there. Either this param or params after this are required to
            construct a sampler.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.
    """

    def __init__(
            self,
            agents,
            envs,
            *,  # After this require passing by keyword.
            worker_factory=None,
            max_episode_length=None,
            is_tf_worker=False,
            seed=get_seed(),
            n_workers=psutil.cpu_count(logical=False),
            worker_class=SurpriseWorker,
            worker_args=None):
        # pylint: disable=super-init-not-called
        if worker_factory is None and max_episode_length is None:
            raise TypeError('Must construct a sampler from WorkerFactory or'
                            'parameters (at least max_episode_length)')
        if isinstance(worker_factory, SurpriseWorkerFactory):
            self._factory = worker_factory
        else:
            self._factory = SurpriseWorkerFactory(
                max_episode_length=max_episode_length,
                is_tf_worker=is_tf_worker,
                seed=seed,
                n_workers=n_workers,
                worker_class=worker_class,
                worker_args=worker_args)

        self._agents = self._factory.prepare_worker_messages(agents)
        self._envs = self._factory.prepare_worker_messages(
            envs, preprocess=copy.deepcopy)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)
        self.total_env_steps = 0

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, envs):
        """Construct this sampler.
        Args:
            worker_factory (WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents (Agent or List[Agent]): Agent(s) to use to sample episodes.
                If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs (Environment or List[Environment]): Environment from which
                episodes are sampled. If a list is passed in, it must have
                length exactly `worker_factory.n_workers`, and will be spread
                across the workers.
        Returns:
            Sampler: An instance of `cls`.
        """
        return cls(agents, envs, worker_factory=worker_factory)

    def _update_workers(self, agent_update, env_update):
        """Apply updates to the workers.
        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        env_updates = self._factory.prepare_worker_messages(
            env_update, preprocess=copy.deepcopy)
        for worker, agent_up, env_up in zip(self._workers, agent_updates,
                                            env_updates):
            worker.update_agent(agent_up)
            worker.update_env(env_up)

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).
        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples (int): Minimum number of transitions / timesteps to
                sample.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        Returns:
            EpisodeBatch: The batch of collected episodes.
        """
        self._update_workers(agent_update, env_update)
        batches = []
        completed_samples = 0
        while True:
            for worker in self._workers:
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                batches.append(batch)
                if completed_samples >= num_samples:
                    samples = EpisodeBatch.concatenate(*batches)
                    self.total_env_steps += sum(samples.lengths)
                    return samples

    def obtain_exact_episodes(self,
                              n_eps_per_worker,
                              agent_update,
                              env_update=None):
        """Sample an exact number of episodes per worker.
        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before samplin episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0,
                then all episodes from worker 1, etc.
        """
        self._update_workers(agent_update, env_update)
        batches = []
        for worker in self._workers:
            for _ in range(n_eps_per_worker):
                batch = worker.rollout()
                batches.append(batch)
        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def shutdown_worker(self):
        """Shutdown the workers."""
        for worker in self._workers:
            worker.shutdown()

    def __getstate__(self):
        """Get the pickle state.
        Returns:
            dict: The pickled state.
        """
        state = self.__dict__.copy()
        # Workers aren't picklable (but WorkerFactory is).
        state['_workers'] = None
        return state

    def __setstate__(self, state):
        """Unpickle the state.
        Args:
            state (dict): Unpickled state.
        """
        self.__dict__.update(state)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)

    def calculate_surprise(self, teacher_returns, student_returns, teacher_new_states, teacher_states_actions):
        teacher_surprise_list = []
        student_surprise_list = []
        for worker in self._workers:
            teacher_surprise, student_surprise = worker.calculate_surprise(teacher_returns, student_returns, teacher_new_states, teacher_states_actions)
            teacher_surprise_list.append(teacher_surprise)
            student_surprise_list.append(student_surprise)

        teacher_surprise = np.concatenate(teacher_surprise_list)
        student_surprise = np.concatenate(student_surprise_list)

        return teacher_surprise, student_surprise
