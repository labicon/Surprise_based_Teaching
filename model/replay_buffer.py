import random
from collections import namedtuple, deque
import numpy as np

import torch

# Define the experience structure
Experience = namedtuple('Experience', ('state_action', 'next_state'))

class ReplayBuffer:
    def __init__(self, buffer_size = 100000, batch_size = 64, device="cpu"):
        """
        Initialize a replay buffer.

        :param buffer_size: int, Maximum number of experiences to store in the buffer
        :param batch_size: int, Number of experiences to sample during a training step
        :param device: str, The device to which tensors should be sent, e.g., "cpu" or "cuda"
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state_action, next_state):
        """Add a new experience to memory."""
        for state_action_i, next_state_i in zip(state_action, next_state):
            e = Experience(state_action_i, next_state_i)
            self.memory.append(e)

    def sample(self, batch_size = None):
        """Randomly sample a batch of experiences from memory."""
        if batch_size is not None:
            self.batch_size = batch_size
            
        if len(self.memory) < self.batch_size:
            experiences = self.memory
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        state_actions = torch.tensor(np.vstack([e.state_action for e in experiences]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.float32).to(self.device)

        return (state_actions, next_states)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
