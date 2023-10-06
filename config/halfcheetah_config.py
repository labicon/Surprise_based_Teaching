'''
Environment Options:
    - SparseHalfCheetah-v2
    - SparseHalfCheetahLimited-v2
    - SparseHalfCheetahAngleLimited-v2
'''
TEACHER_ENV = 'SparseHalfCheetah-v2'
STUDENT_ENV  = 'SparseHalfCheetahAngleLimited-v2'

LOG_DIR  = '../results/Sparse_HalfCheetah_DiffAngle/'

CURRICULUM_WORKER_ARGS = {"surprise": True, "eta0": 0.001, "student_eta0": 0.001, "regressor_hidden_size": 256}

MAX_WORKER_ARGS = {"surprise": True, "eta0": 0.001, "regressor_hidden_size": 256}

MAX_EPISODE_LENGTH = 500
BATCH_SIZE = 5000
N_EPOCHS = 1000