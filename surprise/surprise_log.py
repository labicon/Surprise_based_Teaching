"""Functions exposed directly in the garage namespace."""
from collections import defaultdict
import time

import click
from dowel import tabular
import numpy as np

"""Data types for agent-based learning."""
from dataclasses import dataclass
import enum
from typing import Dict, List
import warnings

import numpy as np

from garage.np import (concat_tensor_dict_list, pad_batch_array,
                       discount_cumsum, stack_tensor_dict_list)

from garage import EpisodeBatch

import torch

def log_surprise(teacher_batch, student_batch, teacher_sampler, discount, prefix='Surprise'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    student_returns = []
    for eps in student_batch.split():
        student_returns.append(discount_cumsum(eps.rewards, discount))

    average_discounted_return = np.mean([rtn[0] for rtn in student_returns])

    teacher_actions = []
    teacher_states = []
    teacher_new_states = []
    teacher_returns = []
    idx = 0
    teacher_surprise_list = []
    student_surprise_list = []
    teacher_surprise_per_step_list = []
    student_surprise_per_step_list = []
    for eps in teacher_batch.split():
        teacher_actions = eps.actions[:len(eps.observations)-1]
        teacher_states = eps.observations[:-1]
        teacher_new_states = eps.observations[1:]
        teacher_returns = discount_cumsum(eps.rewards, discount)

        teacher_states_actions = torch.cat((torch.tensor(teacher_states), torch.tensor(teacher_actions)), dim=1)
        teacher_new_states = torch.tensor(teacher_new_states)

        teacher_surprise, student_surprise = teacher_sampler.calculate_surprise(teacher_returns,
                                                                                student_returns[idx],
                                                                                teacher_new_states,
                                                                                teacher_states_actions)
            
        teacher_surprise_list.append(sum(teacher_surprise))
        student_surprise_list.append(sum(student_surprise))

        teacher_surprise_per_step_list.append(np.mean(teacher_surprise))
        student_surprise_per_step_list.append(np.mean(student_surprise))

        idx += 1

    average_teacher_surprise = np.mean(teacher_surprise_list)
    average_student_surprise = np.mean(student_surprise_list)

    average_teacher_surprise_per_step = np.mean(teacher_surprise_per_step_list)
    average_student_surprise_per_step = np.mean(student_surprise_per_step_list)

    with tabular.prefix(prefix + '/'):
        tabular.record('AverageTeacherSurprise', average_teacher_surprise)
        tabular.record('AverageStudentSurprise', average_student_surprise)
        tabular.record('AverageTeacherSurprisePerStep', average_teacher_surprise_per_step)
        tabular.record('AverageStudentSurprisePerStep', average_student_surprise_per_step)

    return None

