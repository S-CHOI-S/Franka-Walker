import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
# value network가 평가를 대신해 주기 때문에 actor-critic 기반 방법론은 학습할 때 return을 필요로 하지 않음
# 따라서 return을 관측할 때까지 기다릴 필요 없이 하나의 data가 생기면 바로 network를 update
    # data를 어느 정도 모아서 update할 수도 있는데,
    # n_rollout이 몇 개의 data를 모을 지 결정해주는 parameter의 역할을 함

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0): # 정책
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        # prob = np.array(prob, dtype=np.float32)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [] ,[] ,[]
        for transition in self.data:
            s,a,r,s_prime,terminated,truncated = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done = terminated or truncated
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    model = ActorCritic() # model 선언
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s, env_info = env.reset()
        
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(np.array(s, dtype=np.float32))) # model.pi: 정책함수
                m = torch.distributions.Categorical(prob)
                a = m.sample().item()
                s_prime, r, terminated, truncated, info = env.step(a)
                model.put_data((s,a,r,s_prime,terminated,truncated)) # 값들을 잠시 저장

                s = s_prime
                score += r

                if done:
                    break

            model.train_net() # 학습을 진행

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
        
        env.close()

main()