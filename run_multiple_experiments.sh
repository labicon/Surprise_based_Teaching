#!/bin/bash

# First, give the permissions using: chmod +x run_multiple_experiments.sh

# This script runs multiple experiments with different parameters.
NUM_RUNS=3

# config file name
CONFIG="cartpole"

# if you are uisng cartpole, use descrete policy
if [ $CONFIG == "cartpole" ]; then
    for i in $(seq 1 $NUM_RUNS); do
        echo "Run curriculum learning"
        echo "Run $i out of $NUM_RUNS"
        python3 discrete_curriculum.py --config $CONFIG --seed $i

        echo "Run Max surprise learning"
        echo "Run $i out of $NUM_RUNS"
        python3 discrete_max_surprise_trpo.py --config $CONFIG --seed $i
    done
else
    for i in $(seq 1 $NUM_RUNS); do
        echo "Run curriculum learning"
        echo "Run $i out of $NUM_RUNS"
        python3 curriculum.py --config $CONFIG --seed $i

        echo "Run Max surprise learning"
        echo "Run $i out of $NUM_RUNS"
        python3 max_surprise_trpo.py --config $CONFIG --seed $i
    done
fi
