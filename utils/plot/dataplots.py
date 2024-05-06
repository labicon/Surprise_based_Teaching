# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt 
import numpy as np
from numpy import genfromtxt as gen 
import os
import pandas as pd


env_name = 'Sparse_HalfCheetah_Diffspeed'
folder = './' + env_name
Experiment_1_folder = './Sparse_HalfCheetah_Diffspeed/curriculum'
Experiment_2_folder = './Sparse_HalfCheetah_Diffspeed/same_eta/MaxTRPO'

Experiment_1_label = 'Ours'
Experiment_2_label = 'Baseline'

experiment_number = 3

data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(Experiment_1_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(Experiment_1_folder + '_' + str(i) + '/progress.csv'))

# student evaluation
student_evaluation = 'StudentEval/AverageReturn'
# teacher return without surprise
teacher_evaluation = 'TeacherEval/AverageReturn'
termination_rate = 'StudentEval/TerminationRate'

Experiment_1_student_evaluation = []
Experiment_1_teacher_evaluation = []
Experiment_1_termination_rate = []

for i in range(experiment_number):
    Experiment_1_student_evaluation.append(data[i][student_evaluation].to_numpy())
    Experiment_1_teacher_evaluation.append(data[i][teacher_evaluation].to_numpy())
    Experiment_1_termination_rate.append(data[i][termination_rate].to_numpy())


Experiment_1_student_evaluation = np.stack(Experiment_1_student_evaluation) 
Experiment_1_teacher_evaluation = np.stack(Experiment_1_teacher_evaluation)
Experiment_1_termination_rate = np.stack(Experiment_1_termination_rate) 
                                                                                                                                     

Experiment_1_student_evaluation_mean = np.mean(Experiment_1_student_evaluation, axis = 0)
Experiment_1_student_evaluation_std = np.std(Experiment_1_student_evaluation, axis = 0)

Experiment_1_teacher_evaluation_mean = np.mean(Experiment_1_teacher_evaluation, axis = 0)
Experiment_1_teacher_evaluation_std = np.std(Experiment_1_teacher_evaluation, axis = 0)

Experiment_1_termination_mean = np.mean(Experiment_1_termination_rate, axis = 0)
Experiment_1_termination_std = np.std(Experiment_1_termination_rate, axis = 0)


data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(Experiment_2_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(Experiment_2_folder + '_' + str(i) + '/progress.csv'))

Experiment_2_student_evaluation = []
Experiment_2_teacher_evaluation = []
Experiment_2_termination_rate = []

for i in range(experiment_number):
    Experiment_2_student_evaluation.append(data[i][student_evaluation].to_numpy())
    Experiment_2_teacher_evaluation.append(data[i][teacher_evaluation].to_numpy())
    Experiment_2_termination_rate.append(data[i][termination_rate].to_numpy())

Experiment_2_student_evaluation = np.stack(Experiment_2_student_evaluation)
Experiment_2_teacher_evaluation = np.stack(Experiment_2_teacher_evaluation)
Experiment_2_termination_rate = np.stack(Experiment_2_termination_rate)
                                                                                                                                     
Experiment_2_student_evaluation_mean = np.mean(Experiment_2_student_evaluation, axis = 0)
Experiment_2_student_evaluation_std = np.std(Experiment_2_student_evaluation, axis = 0)

Experiment_2_teacher_evaluation_mean = np.mean(Experiment_2_teacher_evaluation, axis = 0)
Experiment_2_teacher_evaluation_std = np.std(Experiment_2_teacher_evaluation, axis = 0)

Experiment_2_termination_mean = np.mean(Experiment_2_termination_rate, axis = 0)
Experiment_2_termination_std = np.std(Experiment_2_termination_rate, axis = 0)

epoch = np.size(Experiment_1_student_evaluation_mean)
iters = np.linspace(0,epoch,epoch)


fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, Experiment_1_student_evaluation_mean, label = Experiment_1_label)
ax1[0].fill_between(iters, (Experiment_1_student_evaluation_mean + Experiment_1_student_evaluation_std), (Experiment_1_student_evaluation_mean - Experiment_1_student_evaluation_std), alpha = .3)

ax1[0].plot(iters, Experiment_2_student_evaluation_mean, label = Experiment_2_label)
ax1[0].fill_between(iters, (Experiment_2_student_evaluation_mean + Experiment_2_student_evaluation_std), (Experiment_2_student_evaluation_mean - Experiment_2_student_evaluation_std), alpha = .3)

ax1[0].set_title(student_evaluation)
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, Experiment_1_teacher_evaluation_mean, label = Experiment_1_label)
ax1[1].fill_between(iters, (Experiment_1_teacher_evaluation_mean + Experiment_1_teacher_evaluation_std), (Experiment_1_teacher_evaluation_mean - Experiment_1_teacher_evaluation_std), alpha = .3)

ax1[1].plot(iters, Experiment_2_teacher_evaluation_mean, label = Experiment_2_label)
ax1[1].fill_between(iters, (Experiment_2_teacher_evaluation_mean + Experiment_2_teacher_evaluation_std), (Experiment_2_teacher_evaluation_mean - Experiment_2_teacher_evaluation_std), alpha = .3)

ax1[1].set_title(teacher_evaluation)
ax1[1].legend()
ax1[1].set_xticks([])


ax1[2].plot(iters, Experiment_1_termination_mean, label = Experiment_1_label)
ax1[2].fill_between(iters, (Experiment_1_termination_mean + Experiment_1_termination_std), (Experiment_1_termination_mean - Experiment_1_termination_std), alpha = .3)
ax1[2].plot(iters, Experiment_2_termination_mean, label = Experiment_2_label)
ax1[2].fill_between(iters, (Experiment_2_termination_mean + Experiment_2_termination_std), (Experiment_2_termination_mean - Experiment_2_termination_std), alpha = .3)
ax1[2].legend()
ax1[2].set_title(termination_rate)

fig1.suptitle(env_name + ': ' + str(experiment_number) + ' experiments')
fig1.savefig('./' + 'diff_speed 3 experiments' + '.png')



