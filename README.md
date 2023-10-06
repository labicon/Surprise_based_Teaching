# surprise-based-curriculum


## Setup config file
Config file should include 
* TEACHER_ENV
    * Name of teacher environment
* STUDENT_ENV
    * Name of student environment
* LOG_DIR
    * Directory that you will save the result
* CURRICULUM_WORKER_ARGS
    * surprise
    * eta0
    * student_eta0
        * It is empty, it will use same eta0 with teacher
    * regressor_hidden_size
        * Dimension of hidden layer in dynamics regressor
* MAX_WORKER_ARGS
    * surprise
    * eta0
    * regressor_hidden_size
        * Dimension of hidden layer in dynamics regressor
* MAX_EPISODE_LENGTH
    * Episode length while training
* BATCH_SIZE
    * Batch size for training
* N_EPOCHS
    * Number of epoch for training

If you add new config file, please add them to curriculum.py, max_surprise_trpo.py, discrete_curriculum.py, and discrete_max_surprise_trpo.py.

## Multiple experiment run
You might use run_multiple_experiment.sh to run multiple experiments. However, you should make them executable before using.

