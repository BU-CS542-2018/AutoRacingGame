'''
Created on Nov 23, 2017

@author: micou
'''
import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Setting some global variables

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
ACTIONS = [[("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", True),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", False),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", True),("KeyEvent", "ArrowRight", False)],\
           [("KeyEvent", "ArrowUp", False),("KeyEvent", "ArrowDown", True),("KeyEvent", "ArrowLeft", False),("KeyEvent", "ArrowRight", True)]]
ACTION_NUM = 3
HIDDEN_SIZE = 120

# Memory replay from pytorch tutorial
class ReplayMemory(object):
    """
    Experience pool for training
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    """
    The model for training
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.hid1 = nn.Linear(1600, HIDDEN_SIZE)
        self.unlf = F.softplus
        self.head = nn.Linear(HIDDEN_SIZE, ACTION_NUM)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(self.unlf(self.hid1(x.view(x.size(0), -1))))
        return x

  
        
        
        
        
        