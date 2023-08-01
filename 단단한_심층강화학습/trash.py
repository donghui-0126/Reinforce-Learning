import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


env = gym.make('CartPole-v1')

state = env.reset()[0]

state = torch.tensor(state, dtype=torch.float32)
print(state)