'''
Environment Options:
    - Sparse_MountainCar-v0
'''

teacher_env = 'Sparse_MountainCar-v0'
student_env = 'Sparse_MountainCar-v0'

log_dir = '../results/Mountaincar/'

curriculum_worker_args = {"surprise": True, "eta0": 0.01, "student_eta0": 0.05, "regressor_hidden_size": 64}

max_worker_args = {"surprise": True, "eta0": 0.01, "regressor_hidden_size": 64}

max_episode_length = 500
batch_size = 1000
n_epochs = 200