'''
Environment Options:
    - SparseHopper-v2
    - SparseHopperLimited-v2
'''

teacher_env = 'SparseHopper-v2'
student_env = 'SparseHopperLimited-v2'

log_dir = '../results/Sparse_Hopper_DiffHeight/'

curriculum_worker_args = {"surprise": True, 
                          "eta0": 0.001, 
                          "student_eta0": 0.001, 
                          "replay": None, 
                          "regressor_hidden_size": 256,
                          "regressor_epoch": 10,
                          "regressor_batch_size": 256}

max_worker_args = {"surprise": True, 
                   "eta0": 0.001, 
                   "replay": None, 
                   "regressor_hidden_size": 256,
                   "regressor_epoch": 10,
                   "regressor_batch_size": 256}

max_episode_length = 500

batch_size = 5000

n_epochs = 1000