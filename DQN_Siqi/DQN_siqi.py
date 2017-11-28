'''
Created on Nov 23, 2017

@author: micou, siqi

Modification:
        changed the parameters for convnet
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
ACTION_NUM = 8
HIDDEN_SIZE = 512

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
        # 3 conv layer
        self.conv1 = nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #FC layer
        self.hid1 = nn.Linear(441, HIDDEN_SIZE)
        self.unlf = F.relu
        self.head = nn.Linear(HIDDEN_SIZE, ACTION_NUM)
        
    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(conv1)))
        conv3 = F.relu(self.bn3(self.conv3(conv2)))
        linear = self.head(self.unlf(self.hid1(conv3.view(conv3.size(0), -1))))
        return x

  
        
        
        
        
        
