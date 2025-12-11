import torch 
import torch as nn
import torch.nn.functional as F
import os
import random
import time
import torch.optim as optim
import torch.utils.tensorboard
from torch_geometric.nn import GATConv

#Implementação do DQN
#obs.shape = (N_CHANNELS, VIEW_SIZE, VIEW_SIZE)
class DQN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        C ,H  ,W = obs_shape
