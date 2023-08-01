import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

seed_value = 42
np.random.seed(seed_value)
seed_value = 42
torch.manual_seed(seed_value)
seed_value = 42
random.seed(seed_value)

gamma = 0.9

class SARSA_Agent(nn.Module):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        super(SARSA_Agent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_states, 32),
            nn.Linear(32, 32),
            nn.Linear(32, num_actions)
        )
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.epsilon = 1.
        self.epsilon_decay = .99995
        self.epsilon_min = 0.01
    
        self.train() # 훈련 모드 설정
        
    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        # 입실론 그리디 정책
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
        return action
    
    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
def train(q_agent, state, action, reward, next_state, next_action, done):
    q_agent.decrease_epsilon()
    q_agent.optimizer.zero_grad()

    q_value = q_agent.model((torch.tensor(state)))[action]
    next_q_value = q_agent.model(torch.tensor(next_state))[next_action].detach()
    # q_value = q_agent.forward(torch.tensor(state))
    # next_q_value = q_agent.forward(torch.tensor(next_state))
    
    # q_value = q_value.gather(-1, torch.tensor(action).unsqueeze(-1)).squeeze(-1)
    # next_q_value = next_q_value.gather(-1, torch.tensor(next_action).unsqueeze(-1)).squeeze(-1)
    
    q_target = reward + (1 - int(done)) * gamma * next_q_value
    

    td_error = (q_target - q_value)**2
    
    # print(q_target, q_value)

    # print("### q_target")
    # print(q_target)
    # print("### q value")
    # print(q_value)
    # print("### loss")
    # print(loss)
    
    td_error.backward()                 # 역전파
    q_agent.optimizer.step()                # 경사 하강, 가중치를 업데이트
    
    return td_error

def main():
    env = gym.make('CartPole-v1')
    q_agent = SARSA_Agent(4,2)    # SARSA를 위한 행동-가치 함수
    
    max_epi = 500
    rewards = []

    for epi in range(max_epi):
        state = torch.tensor(env.reset()[0])
        done = False
        total_reward = 0
        action = q_agent.act(state)
        
        
        ep_rewards = 0
        losses = []
        while not done:
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward

            next_action = q_agent.act(torch.tensor(next_state))

            # 훈련을 위해 데이터 저장
            loss = train(q_agent, state, action, reward, next_state, next_action, done)
            losses.append(loss)

            # print(action, reward, next_state, next_action, done)
            
            ep_rewards += reward
            state = next_state
            action = next_action
                        
        rewards.append(ep_rewards)
        ep_loss = sum(losses) / len(losses)
        
        if (epi+1) % 10 == 0:
            # print(losses)
            print("episode: {}, eps: {:.3f}, loss: {:.1f}, rewards: {}".format(epi+1, q_agent.epsilon, ep_loss, ep_rewards))
            
if __name__ == '__main__':
    main()
