# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:11:12 2023

@author: clemm
"""

import matplotlib.pyplot as plt 
import numpy as np
from numpy import genfromtxt as gen 
import os



ii = 4
jj = 29
kk = 1

st1 = gen('./Sparse_HalfCheetah/curriculum/progress.csv', delimiter=',')
st2 = gen('./Sparse_HalfCheetah/curriculum_1/progress.csv', delimiter=',')
st3 = gen('./Sparse_HalfCheetah/curriculum_2/progress.csv', delimiter=',')
# st4 = gen('./Sparse_HalfCheetah/curriculum_3/progress.csv', delimiter=',')
# st5 = gen('./Sparse_HalfCheetah/curriculum_4/progress.csv', delimiter=',')



student_c = np.stack([st1[1:, ii], st2[1:, ii], st3[1:, ii]]) #, st4[1:, ii], st5[1:, ii]]) #, st6[1:, ii]]) # , st7[1:, ii], st8[1:, ii],st9[1:, ii]])
teacher_surp_c = np.stack([st1[1:, jj], st2[1:, jj], st3[1:, jj]]) #, st4[1:, jj], st5[1:, jj]]) # , st6[1:, jj]] ) #, st7[1:, jj], st8[1:, jj],st9[1:, jj]])
teacher_c = np.stack([st1[1:, kk], st2[1:, kk], st3[1:, kk]]) #, st4[1:, kk], st5[1:, kk]])# , st6[1:, kk]] )#, st7[1:, kk], st8[1:, kk],st9[1:, kk]])
                                                                                                                                     

s_mean_c = np.mean(student_c, axis = 0)
s_std_c = np.std(student_c, axis = 0)

ts_mean_c = np.mean(teacher_surp_c, axis = 0)
ts_std_c = np.std(teacher_surp_c, axis = 0)

t_mean_c = np.mean(teacher_c, axis = 0)
t_std_c = np.std(teacher_c, axis = 0)

ii = 4
jj = 29
kk = 1

st1 = gen('./Sparse_HalfCheetah/MaxTRPO_4/progress.csv', delimiter=',')
st2 = gen('./Sparse_HalfCheetah/MaxTRPO_5/progress.csv', delimiter=',')
st3 = gen('./Sparse_HalfCheetah/MaxTRPO_6/progress.csv', delimiter=',')
# st4 = gen('./Sparse_HalfCheetah/MaxTRPO_3/progress.csv', delimiter=',')
# st5 = gen('./Sparse_HalfCheetah/MaxTRPO_4/progress.csv', delimiter=',')

student_m = np.stack([st1[1:, ii], st2[1:, ii], st3[1:, ii]]) #, st4[1:, ii], st5[1:, ii]]) #, st6[1:, ii]])
teacher_surp_m = np.stack([st1[1:, jj], st2[1:, jj], st3[1:, jj]]) #, st4[1:, jj], st5[1:, jj]]) # , st6[1:, jj] ])
teacher_m = np.stack([st1[1:, kk], st2[1:, kk], st3[1:, kk]]) #, st4[1:, kk], st5[1:, kk]]) #, st6[1:, kk] ])
                                                                                                                                     

s_mean_m = np.mean(student_m, axis = 0)
s_std_m = np.std(student_m, axis = 0)

ts_mean_m = np.mean(teacher_surp_m, axis = 0)
ts_std_m = np.std(teacher_surp_m, axis = 0)

t_mean_m = np.mean(teacher_m, axis = 0)
t_std_m = np.std(teacher_m, axis = 0)


# ii = 27
# jj = 19
# kk = 14

# st1 = gen('./Swimmer_0001/curriculum/progress.csv', delimiter=',')
# st2 = gen('./Swimmer_0001/curriculum_1/progress.csv', delimiter=',')
# st3 = gen('./Swimmer_0001/curriculum_2/progress.csv', delimiter=',')
# st4 = gen('./Swimmer_0001/curriculum_3/progress.csv', delimiter=',')
# st5 = gen('./Swimmer_0001/curriculum_4/progress.csv', delimiter=',')

# student_c_0001 = np.stack([st1[1:, ii], st2[1:, ii] , st3[1:, ii], st4[1:, ii], st5[1:, ii]]) #, st6[1:, ii]])
# teacher_surp_c_0001 = np.stack([st1[1:, jj], st2[1:, jj] , st3[1:, jj], st4[1:, jj], st5[1:, jj]]) # , st6[1:, jj] ])
# teacher_c_0001 = np.stack([st1[1:, kk], st2[1:, kk] , st3[1:, kk], st4[1:, kk], st5[1:, kk]]) #, st6[1:, kk] ])
                                                                                                                                     

# s_mean_c_0001 = np.mean(student_c_0001, axis = 0)
# s_std_c_0001 = np.std(student_c_0001, axis = 0)

# ts_mean_c_0001 = np.mean(teacher_surp_c_0001, axis = 0)
# ts_std_c_0001 = np.std(teacher_surp_c_0001, axis = 0)

# t_mean_c_0001 = np.mean(teacher_c_0001, axis = 0)
# t_std_c_0001 = np.std(teacher_c_0001, axis = 0)




iters = np.linspace(0,1000,1000)


fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, s_mean_c, label = 'ours_eta_0.001')
ax1[0].fill_between(iters, (s_mean_c + s_std_c), (s_mean_c - s_std_c), alpha = .3)

ax1[0].plot(iters, s_mean_m, label = 'max surp_eta_0.001.')
ax1[0].fill_between(iters, (s_mean_m + s_std_m), (s_mean_m - s_std_m), alpha = .3)

# ax1[0].plot(iters, s_mean_c_0001, label = 'ours_eta_0.001')
# ax1[0].fill_between(iters, (s_mean_c_0001 + s_std_c_0001), (s_mean_c_0001 - s_std_c_0001), alpha = .3)

ax1[0].set_title('student')
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, t_mean_c, label = 'ours_eta_0.001')
ax1[1].fill_between(iters, (t_mean_c + t_std_c), (t_mean_c - t_std_c), alpha = .3)

ax1[1].plot(iters, t_mean_m, label = 'max surp_eta_0.001.')
ax1[1].fill_between(iters, (t_mean_m + t_std_m), (t_mean_m - t_std_m), alpha = .3)

# ax1[1].plot(iters, t_mean_c_0001, label = 'ours_eta_0.001')
# ax1[1].fill_between(iters, (t_mean_c_0001 + t_std_c_0001), (t_mean_c_0001 - t_std_c_0001), alpha = .3)

ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title('teacher (extrinsic)')

ax1[2].plot(iters, ts_mean_c, label = 'ours_eta_0.001')
ax1[2].fill_between(iters, (ts_mean_c + ts_std_c), (ts_mean_c - ts_std_c), alpha = .3)
ax1[2].plot(iters, ts_mean_m, label = 'max surp_eta_0.001.')
ax1[2].fill_between(iters, (ts_mean_m + ts_std_m), (ts_mean_m - ts_std_m), alpha = .3)
# ax1[2].plot(iters, ts_mean_c_0001, label = 'ours_eta_0.001')
# ax1[2].fill_between(iters, (ts_mean_c_0001 + ts_std_c_0001), (ts_mean_c_0001 - ts_std_c_0001), alpha = .3)
ax1[2].set_ylim([-5000000, 5000000])
ax1[2].legend()
ax1[2].set_title('teacher (surprise bonus)')
#ax1[2].set_ylim([-50, 200])

fig1.suptitle('Sparse HalfCheetah')
fig1.savefig('./Sparse HalfCheetah.png')



'''
ii = 14
jj = 11 
kk =  23

st1 = gen('./CartPole/curriculum_diff/progress.csv', delimiter=',')
st2 = gen('./CartPole/curriculum_diff_1/progress.csv', delimiter=',')
st3 = gen('./CartPole/curriculum_diff_2/progress.csv', delimiter=',')



student_dc = np.stack([st1[1:, ii] , st2[1:, ii] , st3[1:, ii]]) #, st4[1:, ii], st5[1:, ii] , st6[1:, ii]]) # , st7[1:, ii], st8[1:, ii],st9[1:, ii]])
teacher_surp_dc = np.stack([st1[1:, jj] , st2[1:, jj] , st3[1:, jj]]) #, st4[1:, jj], st5[1:, jj] , st6[1:, jj]] ) #, st7[1:, jj], st8[1:, jj],st9[1:, jj]])
teacher_dc = np.stack([st1[1:, kk] , st2[1:, kk] , st3[1:, kk]]) #, st4[1:, kk], st5[1:, kk] , st6[1:, kk]] )#, st7[1:, kk], st8[1:, kk],st9[1:, kk]])
                                                                                                                                     

s_mean_dc = np.mean(student_dc, axis = 0)
s_std_dc = np.std(student_dc, axis = 0)

ts_mean_dc = np.mean(teacher_surp_dc, axis = 0)
ts_std_dc = np.std(teacher_surp_dc, axis = 0)

t_mean_dc = np.mean(teacher_dc, axis = 0)
t_std_dc = np.std(teacher_dc, axis = 0)

ii = 14
jj = 11 
kk = 23

st1 = gen('./CartPole/MaxTRPO_diff/progress.csv', delimiter=',')
st2 = gen('./CartPole/MaxTRPO_diff_1/progress.csv', delimiter=',')
st3 = gen('./CartPole/MaxTRPO_diff_2/progress.csv', delimiter=',')


student_dm = np.stack([st1[1:, ii] , st2[1:, ii] , st3[1:, ii]]) #, st4[1:, ii], st5[1:, ii] , st6[1:, ii]])
teacher_surp_dm = np.stack([st1[1:, jj] , st2[1:, jj] , st3[1:, jj]]) #, st4[1:, jj], st5[1:, jj] , st6[1:, jj] ])
teacher_dm = np.stack([st1[1:, kk] , st2[1:, kk] , st3[1:, kk]]) #, st4[1:, kk], st5[1:, kk] , st6[1:, kk] ])
                                                                                                                                     

s_mean_dm = np.mean(student_dm, axis = 0)
s_std_dm = np.std(student_dm, axis = 0)

ts_mean_dm = np.mean(teacher_surp_dm, axis = 0)
ts_std_dm = np.std(teacher_surp_dm, axis = 0)

t_mean_dm = np.mean(teacher_dm, axis = 0)
t_std_dm = np.std(teacher_dm, axis = 0)




iters = np.linspace(0,200,200)


fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, s_mean_dc, label = 'ours')
ax1[0].fill_between(iters, (s_mean_dc + s_std_dc), (s_mean_dc - s_std_dc), alpha = .3)

ax1[0].plot(iters, s_mean_dm, label = 'max surp.')
ax1[0].fill_between(iters, (s_mean_dm + s_std_dm), (s_mean_dm - s_std_dm), alpha = .3)

ax1[0].set_title('student')
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, t_mean_dc, label = 'ours')
ax1[1].fill_between(iters, (t_mean_dc + t_std_dc), (t_mean_dc - t_std_dc), alpha = .3)

ax1[1].plot(iters, t_mean_dm, label = 'max surp.')
ax1[1].fill_between(iters, (t_mean_dm + t_std_dm), (t_mean_dm - t_std_dm), alpha = .3)

ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title('teacher (extrinsic)')

ax1[2].plot(iters, ts_mean_dc, label = 'ours')
ax1[2].fill_between(iters, (ts_mean_dc + ts_std_dc), (ts_mean_dc - ts_std_dc), alpha = .3)
ax1[2].plot(iters, ts_mean_dm, label = 'max surp.')
ax1[2].fill_between(iters, (ts_mean_dm + ts_std_dm), (ts_mean_dm - ts_std_dm), alpha = .3)
#ax1[2].plot(iters, s_mean_m, label = 'max surp.')
#ax1[2].fill_between(iters, (s_mean_m + s_std_m), (s_mean_m - s_std_m), alpha = .3)
ax1[2].legend()
ax1[2].set_title('teacher (surprise bonus)')
#ax1[2].set_ylim([-50, 200])

fig1.suptitle('CartPole - Diff. Constraints')
fig1.savefig('./Cartpole_diff_exp.png')



fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, s_mean_c[:200], label = 'standard')
ax1[0].fill_between(iters, (s_mean_c[:200] + s_std_c[:200]), (s_mean_c[:200] - s_std_c[:200]), alpha = .3)

ax1[0].plot(iters, s_mean_dc, label = 'diff. const.')
ax1[0].fill_between(iters, (s_mean_dc + s_std_dc), (s_mean_dc - s_std_dc), alpha = .3)

ax1[0].set_title('student')
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, t_mean_c[:200], label = 'standard')
ax1[1].fill_between(iters, (t_mean_c[:200] + t_std_c[:200]), (t_mean_c[:200] - t_std_c[:200]), alpha = .3)

ax1[1].plot(iters, t_mean_dc, label = 'diff. const.')
ax1[1].fill_between(iters, (t_mean_dc + t_std_dc), (t_mean_dc - t_std_dc), alpha = .3)

ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title('teacher (extrinsic)')

ax1[2].plot(iters, ts_mean_c[:200], label = 'standard')
ax1[2].fill_between(iters, (ts_mean_c[:200] + ts_std_c[:200]), (ts_mean_c[:200] - ts_std_c[:200]), alpha = .3)
ax1[2].plot(iters, ts_mean_dc, label = 'diff. const.')
ax1[2].fill_between(iters, (ts_mean_dc + ts_std_dc), (ts_mean_dc - ts_std_dc), alpha = .3)
#ax1[2].plot(iters, s_mean_m, label = 'max surp.')
#ax1[2].fill_between(iters, (s_mean_m + s_std_m), (s_mean_m - s_std_m), alpha = .3)
ax1[2].legend()
ax1[2].set_title('teacher (surprise bonus)')
#ax1[2].set_ylim([-50, 200])

fig1.suptitle('CartPole- Std env vs. diff constraints (our method only)')
fig1.savefig('./Cartpole_diff_exp_ours.png')



fig1, ax1 = plt.subplots(3, figsize = (10,10))
ax1[0].plot(iters, s_mean_m[:200], label = 'standard')
ax1[0].fill_between(iters, (s_mean_m[:200] + s_std_m[:200]), (s_mean_m[:200] - s_std_m[:200]), alpha = .3)

ax1[0].plot(iters, s_mean_dm, label = 'diff. const.')
ax1[0].fill_between(iters, (s_mean_dm + s_std_dm), (s_mean_dm - s_std_dm), alpha = .3)

ax1[0].set_title('student')
ax1[0].legend()
ax1[0].set_xticks([])

ax1[1].plot(iters, t_mean_m[:200], label = 'standard')
ax1[1].fill_between(iters, (t_mean_m[:200] + t_std_m[:200]), (t_mean_m[:200] - t_std_m[:200]), alpha = .3)

ax1[1].plot(iters, t_mean_dm, label = 'diff. const.')
ax1[1].fill_between(iters, (t_mean_dm + t_std_dm), (t_mean_dm - t_std_dm), alpha = .3)

ax1[1].set_xticks([])
ax1[1].legend()
ax1[1].set_title('teacher (extrinsic)')

ax1[2].plot(iters, ts_mean_m[:200], label = 'standard')
ax1[2].fill_between(iters, (ts_mean_m[:200] + ts_std_m[:200]), (ts_mean_m[:200] - ts_std_m[:200]), alpha = .3)
ax1[2].plot(iters, ts_mean_dm, label = 'diff. const.')
ax1[2].fill_between(iters, (ts_mean_dm + ts_std_dm), (ts_mean_dm - ts_std_dm), alpha = .3)
#ax1[2].plot(iters, s_mean_m, label = 'max surp.')
#ax1[2].fill_between(iters, (s_mean_m + s_std_m), (s_mean_m - s_std_m), alpha = .3)
ax1[2].legend()
ax1[2].set_title('teacher (surprise bonus)')
#ax1[2].set_ylim([-50, 200])

fig1.suptitle('CartPole- Std env vs. diff constraints (max surp. only)')
fig1.savefig('./Cartpole_diff_exp_max.png')
'''