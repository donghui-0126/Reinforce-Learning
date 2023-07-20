import numpy as np
import pandas as pd
import random
from collections import defaultdict
import gym

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import random

seed_value = 42
np.random.seed(seed_value)
seed_value = 42
torch.manual_seed(seed_value)
seed_value = 42
random.seed(seed_value)


class DeepSARSA:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = 0.001
        self.gamma = 0.99

        self.model = nn.Sequential(
            nn.Linear(self.num_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.epsilon = 1.
        self.epsilon_decay = .99995
        self.epsilon_min = 0.01 
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            
        return action
    
    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update(self, state, action, reward, next_state, next_action, done):
        self.decrease_epsilon()
        self.optimizer.zero_grad()
        
        q_value = self.model((state))[action]
        next_q_value = self.model(next_state)[next_action].detach()
        q_target = reward + (1 - int(done)) * self.gamma * next_q_value
        q_error = (q_target - q_value) ** 2
        
        # print(q_target, q_value)
        
        # print("### q_target")
        # print(q_target)
        # print("### q value")
        # print(q_value)

        
        q_error.backward()
        self.optimizer.step()
        
        return q_error.item()
    
    
import gym
from gym import wrappers
env = gym.make('CartPole-v1')
agent = DeepSARSA(4,2)

rewards = []
for ep in range(500):
    done = False
    obs = torch.tensor((env.reset())[0])
    action = agent.act(obs)

    ep_rewards = 0
    losses = []
    while not done:
        next_obs, reward, done, info, _ = env.step((action))
        next_obs = torch.FloatTensor(next_obs)

        next_action = agent.act((next_obs))
        
        loss = agent.update(obs, action, reward, next_obs, next_action, done)
        
        losses.append(loss)
        
        # print(action, reward, next_obs, next_action, done)

        ep_rewards += reward
        obs = next_obs
        action = next_action
        
    rewards.append(ep_rewards)
    ep_loss = sum(losses) / len(losses)
    if (ep+1) % 10 == 0:
        
        print("episode: {}, eps: {:.3f}, loss: {:.1f}, rewards: {}".format(ep+1, agent.epsilon, ep_loss, ep_rewards))
env.close()