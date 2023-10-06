from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
import torch
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed

from surprise.SurpriseFunctions import SurpriseWorkerFactory, CustomSampler, SurpriseWorker

from algo.StudentTeacherAlgo_Diff import Curriculum_Diff as Curriculum 
from surprise.trainer_diff import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='halfcheetah', type=str, help='name of config file')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.config == 'halfcheetah':
    from config.halfcheetah_config import *
elif args.config == 'hopper':
    from config.hopper_config import *
elif args.config == 'cartpole':
    raise ValueError("Invalid config")
elif args.config == 'mountaincar':
    from config.mountaincar_config import *
else:
    raise ValueError("Invalid config")

########################################################################################
# If you choose snapshot_mode='all', you will save every iteration of the experiment.
@wrap_experiment(log_dir = LOG_DIR + 'curriculum', archive_launch_repo=False)
def curriculum_student_teacher(ctxt=None, seed=1):
    """
    Student-teacher setup for auto-curricula using both teacher and student surprise 
    """
    set_seed(seed)
    teacher_env = GymEnv(TEACHER_ENV)
    student_env = GymEnv(STUDENT_ENV)

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
                           max_episode_length = MAX_EPISODE_LENGTH)
    
    surprise = {"surprise": CURRICULUM_WORKER_ARGS ["surprise"], 
                "student": student_policy, 
                "eta0": CURRICULUM_WORKER_ARGS ["eta0"], 
                "student_eta0": CURRICULUM_WORKER_ARGS ["student_eta0"], 
                "replay": student_sampler,
                "regressor_hidden_size": CURRICULUM_WORKER_ARGS ["regressor_hidden_size"],
                "regressor_epoch": CURRICULUM_WORKER_ARGS ["regressor_epoch"],
                "regressor_batch_size": CURRICULUM_WORKER_ARGS ["regressor_batch_size"],
                "state_dim": teacher_env.spec.observation_space.flat_dim,
                "action_dim": teacher_env.spec.action_space.flat_dim}
    
    teacher_sampler = CustomSampler(envs = teacher_env,  
                           agents = teacher_policy, 
                           worker_factory = SurpriseWorkerFactory,
                           worker_class = SurpriseWorker, 
                           worker_args = surprise, 
                           max_episode_length = MAX_EPISODE_LENGTH)
    
    
    algo = Curriculum(teacher_env_spec = teacher_env.spec,
                      student_env_spec = student_env.spec,
                      teacher_policy = teacher_policy,
                      teacher_value_function = value_function, 
                      student_policy = student_policy, 
                      teacher_sampler = teacher_sampler,
                      student_sampler = student_sampler,
                      batch_size = BATCH_SIZE )
    
    trainer.setup( algo = algo, env = teacher_env, student_env=student_env)
    trainer.train(n_epochs = N_EPOCHS, batch_size = BATCH_SIZE , store_episodes=True)

curriculum_student_teacher(seed = args.seed)