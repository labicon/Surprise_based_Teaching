'''
Environment Options:
    - CartPoleSwingup-v0
'''

TEACHER_ENV = 'CartPoleSwingup-v0'
STUDENT_ENV = 'CartPoleSwingup-v0'

LOG_DIR = '../results/CartpoleSwingup/'

CURRICULUM_WORKER_ARGS = {"surprise": True, 
                          "eta0": 0.005, 
                          "student_eta0": 0.005, 
                          "regressor_hidden_size": 64,
                          "regressor_epoch": 10,
                          "regressor_batch_size": 256}

MAX_WORKER_ARGS = {"surprise": True, 
                   "eta0": 0.005, 
                   "regressor_hidden_size": 64,
                   "regressor_epoch": 10,
                   "regressor_batch_size": 256}

MAX_EPISODE_LENGTH = 500
BATCH_SIZE = 1000
N_EPOCHS = 200