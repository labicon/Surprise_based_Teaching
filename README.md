# surprise-based-curriculum

This is the official code for [Adaptive Teaching in Heterogeneous Agents: Balancing Surprise in Sparse Reward Scenarios](https://arxiv.org/abs/2405.14199), published in 2024 Learning for Dynamics and Control Conference (L4DC).

## How to run multiple experiments

1. Register environment to gym
    * Sample environment code is given in `env/`
    * If you are planning to use different constraint for teacher and student, please register them as different environment.
2. Change hyperparameter in config file
    * If you want to add config file for new environment, please change `curriculum.py, discrete_curriculum.py, max_surprise_trpo.py, and discrete_max_surprise_trpo.py` accordingly.
3. Make `run_multiple_experiments.sh` executable
```bash
chmod +x run_multiple_experiments.sh
```
4. Run bash file
```bash
bash run_multiple_experiments.sh
```

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
    * regressor_epoch
        * Training epoch for regressor model
    * regressor_batch_size
        * Training batch size for regressor model
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

