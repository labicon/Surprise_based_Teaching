# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np
from numpy import genfromtxt as gen 
import os
import pandas as pd


env_name = 'CartpoleSwingup'
folder = './' + env_name
PPO_folder = folder + '/PPO_baseline'
TRPO_folder = folder + '/TRPO_baseline'

experiment_number = 3

data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(PPO_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(PPO_folder + '_' + str(i) + '/progress.csv'))

evaluation = 'Evaluation/AverageReturn'
termination_rate = 'Evaluation/TerminationRate'

PPO_evaluation = []
PPO_termination_rate = []

for i in range(experiment_number):
    PPO_evaluation.append(data[i][evaluation].to_numpy())
    PPO_termination_rate.append(data[i][termination_rate].to_numpy())


PPO_evaluation = np.stack(PPO_evaluation) 
PPO_termination_rate = np.stack(PPO_termination_rate) 
                                                                                                                                     

PPO_evaluation_mean = np.mean(PPO_evaluation, axis = 0)
PPO_evaluation_std = np.std(PPO_evaluation, axis = 0)

PPO_termination_mean = np.mean(PPO_termination_rate, axis = 0)
PPO_termination_std = np.std(PPO_termination_rate, axis = 0)


data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(TRPO_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(TRPO_folder + '_' + str(i) + '/progress.csv'))

TRPO_evaluation = []
TRPO_termination_rate = []

for i in range(experiment_number):
    TRPO_evaluation.append(data[i][evaluation].to_numpy())
    TRPO_termination_rate.append(data[i][termination_rate].to_numpy())

TRPO_evaluation = np.stack(TRPO_evaluation)
TRPO_termination_rate = np.stack(TRPO_termination_rate)
                                                                                                                                     
TRPO_evaluation_mean = np.mean(TRPO_evaluation, axis = 0)
TRPO_evaluation_std = np.std(TRPO_evaluation, axis = 0)

TRPO_termination_mean = np.mean(TRPO_termination_rate, axis = 0)
TRPO_termination_std = np.std(TRPO_termination_rate, axis = 0)

epoch = np.size(PPO_evaluation_mean)
iters = np.linspace(0,epoch,epoch)


fig1, ax1 = plt.subplots(2, figsize = (10,10))
ax1[0].plot(iters, PPO_evaluation_mean, label = 'PPO')
ax1[0].fill_between(iters, (PPO_evaluation_mean + PPO_evaluation_std), (PPO_evaluation_mean - PPO_evaluation_std), alpha = .3)

ax1[0].plot(iters, TRPO_evaluation_mean, label = 'TRPO.')
ax1[0].fill_between(iters, (TRPO_evaluation_mean + TRPO_evaluation_std), (TRPO_evaluation_mean - TRPO_evaluation_std), alpha = .3)

ax1[0].set_title(evaluation)
ax1[0].legend()
ax1[0].set_xticks([])


ax1[1].plot(iters, PPO_termination_mean, label = 'PPO')
ax1[1].fill_between(iters, (PPO_termination_mean + PPO_termination_std), (PPO_termination_mean - PPO_termination_std), alpha = .3)
ax1[1].plot(iters, TRPO_termination_mean, label = 'TRPO.')
ax1[1].fill_between(iters, (TRPO_termination_mean + TRPO_termination_std), (TRPO_termination_mean - TRPO_termination_std), alpha = .3)
ax1[1].legend()
ax1[1].set_title(termination_rate)

fig1.suptitle(env_name + ': ' + str(experiment_number) + ' experiments')
fig1.savefig('./' + env_name + '.png')



