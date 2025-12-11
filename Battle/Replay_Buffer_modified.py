import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done, edge_index):
        self.buffer.append((state, action, reward, next_state, done, edge_index))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, graphs = zip(*batch)

        # Converter tudo para tensores PyTorch
        states      = torch.tensor(states, dtype=torch.float32)
        actions     = torch.tensor(actions, dtype=torch.long)
        rewards     = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones       = torch.tensor(dones, dtype=torch.float32)


        return states, actions, rewards, next_states, dones, graphs

    def __len__(self):
        return len(self.buffer)

    def erase(self):
        self.buffer.clear()