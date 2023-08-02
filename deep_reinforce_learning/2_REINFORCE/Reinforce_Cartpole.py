from torch.distributions import Categorical 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # 훈련 모드 설정
        
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
        
    def forward(self, x):
        pdparam = self.model(x)
        return pdparam
    
    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # 텐서로 변경
        pdparam = self.forward(x)                      # 전방 전달
        pd = Categorical(logits=pdparam)               # 확률 분포
        action = pd.sample()                           # 확률 분포를 통한 행동 정책
        log_prob = pd.log_prob(action)                 # 
        self.log_probs.append(log_prob)                # 훈련을 위해 저장
        return action.item()
    
def train(pi, optimizer):
    # REINFORCE 알고리즘의 내부 경사 상승 루프
    T = len(pi.rewards) 
    rets = np.empty(T, dtype=np.float32)        # 이득
    future_ret = 0.0
    # 이득을 효율적으로 계산
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    print(rets)
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets        # 경사 항, 최대화를 위해서 음의 부호로 함
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()                 # 역전파
    optimizer.step()                # 경사 상승, 가중치를 업데이트
    return loss

def main():
    env = gym.make('CartPole-v0',  render_mode="human")
    in_dim = env.observation_space.shape[0] # 4
    out_dim = env.action_space.n            # 2
    pi = Pi(in_dim, out_dim)                # REINFORCE를 위한 정책 
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(300):
        state = env.reset()
        for t in range(200):                    
            if t == 0:
                action = pi.act(state[0])
            else:
                action = pi.act(state)
            
            state, reward, done, _,__ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
            
        loss = train(pi, optimizer)
                
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f'Episode {epi}, loss: {loss}, \
            total_reward: {total_reward}, solved: {solved}')
            
if __name__ == '__main__':
    main()