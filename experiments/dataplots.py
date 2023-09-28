# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:11:12 2023

@author: clemm
"""

import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Sparse_HalfCheetah_Diffspeed')
parser.add_argument('--experiment_number', type=int, default=1)

args = parser.parse_args()

env_name = args.env
folder = './' + env_name
curriculum_folder = folder + '/curriculum'
MaxTRPO_folder = folder + '/MaxTRPO'

experiment_number = args.experiment_number

data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(curriculum_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(curriculum_folder + '_' + str(i) + '/progress.csv'))

# student evaluation
student_evaluation = 'StudentEval/AverageReturn'
# teacher return with surprise
teacher_w_surprise = 'StudentEval/TerminationRate'
# teacher return without surprise
teacher_wout_surprise = 'TeacherEval/AverageReturn'

curriculum_student = []
curriculum_teacher_surp = []
curriculum_teacher = []

for i in range(experiment_number):
    curriculum_student.append(data[i][student_evaluation].to_numpy())
    curriculum_teacher_surp.append(data[i][teacher_w_surprise].to_numpy())
    curriculum_teacher.append(data[i][teacher_wout_surprise].to_numpy())

student_c = np.stack(curriculum_student) 
teacher_surp_c = np.stack(curriculum_teacher_surp) 
teacher_c = np.stack(curriculum_teacher)
                                                                                                                                     

s_mean_c = np.mean(student_c, axis = 0)
s_std_c = np.std(student_c, axis = 0)

ts_mean_c = np.mean(teacher_surp_c, axis = 0)
ts_std_c = np.std(teacher_surp_c, axis = 0)

t_mean_c = np.mean(teacher_c, axis = 0)
t_std_c = np.std(teacher_c, axis = 0)

data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(MaxTRPO_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(MaxTRPO_folder + '_' + str(i) + '/progress.csv'))

MaxTRPO_student = []
MaxTRPO_teacher_surp = []
MaxTRPO_teacher = []

for i in range(experiment_number):
    MaxTRPO_student.append(data[i][student_evaluation].to_numpy())
    MaxTRPO_teacher_surp.append(data[i][teacher_w_surprise].to_numpy())
    MaxTRPO_teacher.append(data[i][teacher_wout_surprise].to_numpy())

student_m = np.stack(MaxTRPO_student)
teacher_surp_m = np.stack(MaxTRPO_teacher_surp)
teacher_m = np.stack(MaxTRPO_teacher)
                                                                                                                                     

s_mean_m = np.mean(student_m, axis = 0)
s_std_m = np.std(student_m, axis = 0)

ts_mean_m = np.mean(teacher_surp_m, axis = 0)
ts_std_m = np.std(teacher_surp_m, axis = 0)

t_mean_m = np.mean(teacher_m, axis = 0)
t_std_m = np.std(teacher_m, axis = 0)

epoch = np.size(s_mean_c)
iters = np.linspace(0,epoch,epoch)


fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, s_mean_c, label = 'ours_eta_0.001')
ax1[0].fill_between(iters, (s_mean_c + s_std_c), (s_mean_c - s_std_c), alpha = .3)

ax1[0].plot(iters, s_mean_m, label = 'max surp_eta_0.001.')
ax1[0].fill_between(iters, (s_mean_m + s_std_m), (s_mean_m - s_std_m), alpha = .3)

ax1[0].set_title(student_evaluation)
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, t_mean_c, label = 'ours_eta_0.001')
ax1[1].fill_between(iters, (t_mean_c + t_std_c), (t_mean_c - t_std_c), alpha = .3)

ax1[1].plot(iters, t_mean_m, label = 'max surp_eta_0.001.')
ax1[1].fill_between(iters, (t_mean_m + t_std_m), (t_mean_m - t_std_m), alpha = .3)

ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title(teacher_wout_surprise)

ax1[2].plot(iters, ts_mean_c, label = 'ours_eta_0.001')
ax1[2].fill_between(iters, (ts_mean_c + ts_std_c), (ts_mean_c - ts_std_c), alpha = .3)
ax1[2].plot(iters, ts_mean_m, label = 'max surp_eta_0.001.')
ax1[2].fill_between(iters, (ts_mean_m + ts_std_m), (ts_mean_m - ts_std_m), alpha = .3)
ax1[2].legend()
ax1[2].set_title(teacher_w_surprise)

fig1.suptitle(env_name + ': ' + str(experiment_number) + ' experiments')
fig1.savefig('./' + env_name + '.png')

print("curriculum surprise", np.nanmean(ts_mean_c - t_mean_c))
print("Max Surprise", np.nanmean(ts_mean_m - t_mean_m))


