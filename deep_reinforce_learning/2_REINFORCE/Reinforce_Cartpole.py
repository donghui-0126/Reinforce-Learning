from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import back_test_env


class Agent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Agent, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(64, out_dim)
        self.onpolicy_reset()
        self.train() # 훈련 모드 설정
        self.gamma = 0.999
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x, previous_action):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        input = torch.cat(x,previous_action) # 순환 강화학습이기 때문에, 이전 action을 출력
        x = F.sigmoid(input)
        return x

    def act(self, state, previous_action):
        x = torch.from_numpy(state.astype(np.float32)) # 텐서로 변경
        pdparam = self.forward(x, previous_action)                      # 전방 전달
        pd = Categorical(logits=pdparam)               # 확률 분포
        action = pd.sample()                           # 확률 분포를 통한 행동 정책
        log_prob = pd.log_prob(action)                 #
        self.log_probs.append(log_prob)                # 훈련을 위해 저장
        return action.item()

def train(Agent, optimizer):
    # REINFORCE 알고리즘의 내부 경사 상승 루프
    T = len(Agent.rewards)
    rets = np.empty(T, dtype=np.float32)        # 이득
    future_ret = 0.0
    # 이득을 효율적으로 계산
    for t in reversed(range(T)):
        future_ret = Agent.rewards[t] + Agent.gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(Agent.log_probs)
    loss = - log_probs * rets        # 경사 항, 최대화를 위해서 음의 부호로 함
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()                 # 역전파
    optimizer.step()                # 경사 상승, 가중치를 업데이트
    return loss

def main():
    env = back_test_env()
    optimizer = optim.Adam(Agent.parameters(), lr=0.01)
    for epi in range(300):
        state = env.reset()
        for t in range(200):
            previous_action = [0]

            action = Agent.act(state, previous_action[-1])

            state, reward, done, _ = env.step(action)
            Agent.rewards.append(reward)
            env.render()
            if done:
                break

        loss = train(Agent, optimizer)

        total_reward = sum(Agent.rewards)
        solved = total_reward > 195.0
        Agent.onpolicy_reset()

if __name__ == '__main__':
    main()