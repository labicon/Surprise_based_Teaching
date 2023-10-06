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

curriculum_label = 'ours_eta_0.001-0.005'
MaxTRPO_label = 'MaxSurprise_eta_0.001'

save_title = env_name

experiment_number = args.experiment_number

data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(curriculum_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(curriculum_folder + '_' + str(i) + '/progress.csv'))

# student evaluation
student_evaluation = 'StudentEval/AverageReturn'
# teacher return without surprise
teacher_evaluation = 'TeacherEval/AverageReturn'

# Teacher surprise
teacher_surprise = 'Surprise/AverageTeacherSurprise'
# Student surprise
student_surprise = 'Surprise/AverageStudentSurprise'

curriculum_student = []
curriculum_teacher = []

curriculum_teacher_surp = []
curriculum_student_surp = []

for i in range(experiment_number):
    curriculum_student.append(data[i][student_evaluation].to_numpy())
    curriculum_teacher.append(data[i][teacher_evaluation].to_numpy())

    curriculum_teacher_surp.append(data[i][teacher_surprise].to_numpy())
    curriculum_student_surp.append(data[i][student_surprise].to_numpy())

student_c = np.stack(curriculum_student) 
teacher_c = np.stack(curriculum_teacher)

teacher_surp_c = np.stack(curriculum_teacher_surp) 
student_surp_c = np.stack(curriculum_student_surp)                                                                                                                       

# compute mean and std
curriculum_teacher_mean = np.mean(teacher_c, axis = 0)
curriculum_teacher_std = np.std(teacher_c, axis = 0)

curriculum_student_mean = np.mean(student_c, axis = 0)
curriculum_student_std = np.std(student_c, axis = 0)

curriculum_teacher_surp_mean = np.mean(teacher_surp_c, axis = 0)
curriculum_teacher_surp_std = np.std(teacher_surp_c, axis = 0)

curriculum_student_surp_mean = np.mean(student_surp_c, axis = 0)
curriculum_student_surp_std = np.std(student_surp_c, axis = 0)


data = []
for i in range(experiment_number):
    if i == 0:
        data.append(pd.read_csv(MaxTRPO_folder + '/progress.csv'))
    else:
        data.append(pd.read_csv(MaxTRPO_folder + '_' + str(i) + '/progress.csv'))

MaxTRPO_student = []
MaxTRPO_teacher = []

MaxTRPO_teacher_surp = []
MaxTRPO_student_surp = []

for i in range(experiment_number):
    MaxTRPO_student.append(data[i][student_evaluation].to_numpy())
    MaxTRPO_teacher.append(data[i][teacher_evaluation].to_numpy())

    MaxTRPO_teacher_surp.append(data[i][teacher_surprise].to_numpy())
    MaxTRPO_student_surp.append(data[i][student_surprise].to_numpy())

student_m = np.stack(MaxTRPO_student)
teacher_m = np.stack(MaxTRPO_teacher)

teacher_surp_m = np.stack(MaxTRPO_teacher_surp)
student_surp_m = np.stack(MaxTRPO_student_surp)
                                                                                                                                     
# compute mean and std
max_teacher_mean = np.mean(teacher_m, axis = 0)
max_teacher_std = np.std(teacher_m, axis = 0)

max_student_mean = np.mean(student_m, axis = 0)
max_student_std = np.std(student_m, axis = 0)

max_teacher_surprise_mean = np.mean(teacher_surp_m, axis = 0)
max_teacher_surprise_std = np.std(teacher_surp_m, axis = 0)

max_student_surprise_mean = np.mean(student_surp_m, axis = 0)
max_student_surprise_std = np.std(student_surp_m, axis = 0)


# Plotting
epoch = np.size(curriculum_student_mean)
iters = np.linspace(0,epoch,epoch)

fig1, ax1 = plt.subplots(4, figsize = (13,10))
ax1[0].plot(iters, curriculum_student_mean, label = curriculum_label)
ax1[0].fill_between(iters, (curriculum_student_mean + curriculum_student_std), (curriculum_student_mean - curriculum_student_std), alpha = .3)

ax1[0].plot(iters, max_student_mean, label = MaxTRPO_label)
ax1[0].fill_between(iters, (max_student_mean + max_student_std), (max_student_mean - max_student_std), alpha = .3)

ax1[0].set_xticks([])
ax1[0].set_title(student_evaluation)
ax1[0].legend()

# Teacher return
ax1[1].plot(iters, curriculum_teacher_mean, label = curriculum_label)
ax1[1].fill_between(iters, (curriculum_teacher_mean + curriculum_teacher_std), (curriculum_teacher_mean - curriculum_teacher_std), alpha = .3)

ax1[1].plot(iters, max_teacher_mean, label = MaxTRPO_label)
ax1[1].fill_between(iters, (max_teacher_mean + max_teacher_std), (max_teacher_mean - max_teacher_std), alpha = .3)
ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title(teacher_evaluation)

ax1[2].plot(iters, curriculum_teacher_surp_mean, label = curriculum_label)
ax1[2].fill_between(iters, (curriculum_teacher_surp_mean + curriculum_teacher_surp_std), (curriculum_teacher_surp_mean - curriculum_teacher_surp_std), alpha = .3)
ax1[2].plot(iters, max_teacher_surprise_mean, label = MaxTRPO_label)
ax1[2].fill_between(iters, (max_teacher_surprise_mean + max_teacher_surprise_std), (max_teacher_surprise_mean - max_teacher_surprise_std), alpha = .3)
ax1[2].set_xticks([])
ax1[2].legend()
ax1[2].set_title(teacher_surprise)

ax1[3].plot(iters, curriculum_student_surp_mean, label = curriculum_label)
ax1[3].fill_between(iters, (curriculum_student_surp_mean + curriculum_student_surp_std), (curriculum_student_surp_mean - curriculum_student_surp_std), alpha = .3)
ax1[3].plot(iters, max_student_surprise_mean, label = MaxTRPO_label)
ax1[3].fill_between(iters, (max_student_surprise_mean + max_student_surprise_std), (max_student_surprise_mean - max_student_surprise_std), alpha = .3)

ax1[3].legend()
ax1[3].set_title(student_surprise)

fig1.suptitle(env_name + ': ' + str(experiment_number) + ' experiments')
fig1.savefig('./' + save_title + '.png')

print("curriculum surprise", np.nanmean(curriculum_teacher_surp_mean))
print("Max Surprise", np.nanmean(max_teacher_surprise_mean))


