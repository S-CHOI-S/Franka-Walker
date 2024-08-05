import os
import sys
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
from collections import deque

import time
import psutil
import datetime
import subprocess
# import torch
import torchvision
from tensorboard import program
import webbrowser
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from color_code import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#learning rate backward propagation NN action
lr_actor = 0.0003
#learning rate backward propagation NN state value estimation
lr_critic = 0.0003
#Number of Learning Iteration we want to perform
Iter = 100000
#Number max of step to realise in one episode. 
MAX_STEP = 1000
#How rewards are discounted.
gamma = 0.98
#How do we stabilize variance in the return computation.
lambd = 0.95
#batch to train on
batch_size = 64
# Do we want high change to be taken into account.
epsilon = 0.2
#weight decay coefficient in ADAM for state value optim.
l2_rate = 0.001

save_freq = 100

save_flag = False

# Actor class: Used to choose actions of a continuous action space.
class Actor(nn.Module):
    def __init__(self, N_S, N_A, chkpt_dir):
        # Initialize NN structure.
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.sigma = nn.Linear(64,N_A)
        self.mu = nn.Linear(64,N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # This approach use gaussian distribution to decide actions. Could be
        # something else.
        self.distribution = torch.distributions.Normal
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_actor')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def set_init(self,layers):
        # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,state):
        # Use of tanh activation function is recommanded : bounded [-1,1],
        # gives some non-linearity, and tends to give some stability.
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        # mu action output of the NN.
        mu = self.mu(x)
        #log_sigma action output of the NN
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,state):
        # Choose action in the continuous action space using normal distribution
        # defined by mu and sigma of each actions returned by the NN.
        state = torch.from_numpy(np.array(state).astype(np.float32)).unsqueeze(0).to(self.device)
        mu,sigma = self.forward(state)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().cpu().numpy().squeeze(0)
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_model(self, load_model_dir=None):
        if load_model_dir is None:
            load_model_dir = self.checkpoint_file
        self.load_state_dict(torch.load(load_model_dir))
          

# Critic class : Used to estimate V(state) the state value function through a NN.
class Critic(nn.Module):
    def __init__(self, N_S, chkpt_dir):
        # Initialize NN structure.
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)
        self.fc3.weight.data.mul_(0.1) # 초기 weight에 0.1을 곱해주면서 학습을 더 안정적으로 할 수 있도록(tanh, sigmoid를 사용할 경우 많이 쓰는 방식)
        self.fc3.bias.data.mul_(0.0) # bias tensor의 모든 원소를 0으로 설정
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_critic')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def set_init(self,layers):
      # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,state):
      # Use of tanh activation function is recommanded.
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_model(self, load_model_dir=None):
        if load_model_dir is None:
            load_model_dir = self.checkpoint_file
        self.load_state_dict(torch.load(load_model_dir))
    
# Multihead Cost Value Function
class MultiheadCostValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, chkpt_dir=None):
        super(MultiheadCostValueFunction, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])

        self._init_weights()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_multihead')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.shared_layers(x)
        return torch.cat([head(x) for head in self.heads], dim=1)
    
    def _init_weights(self):
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(0.1)
                layer.bias.data.mul_(0.0)
        for head in self.heads:
            head.weight.data.mul_(0.1)
            head.bias.data.mul_(0.0)

    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self, load_model_dir=None):
        if load_model_dir is None:
            load_model_dir = self.checkpoint_file
        self.load_state_dict(torch.load(load_model_dir))

# PPO Algorithm with Constraints   
class PPO:
    def __init__(self, N_S, N_A, log_dir, num_avg_constraints=4, avg_cstrnt_limit=None):
        self.log_dir = log_dir
        
        self.actor_net = Actor(N_S, N_A, log_dir)
        self.critic_net = Critic(N_S, log_dir)
        self.multihead_net = MultiheadCostValueFunction(N_S, 64, num_avg_constraints, chkpt_dir=log_dir)
        
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=1e-3, weight_decay=1e-3)
        self.multihead_optim = optim.Adam(self.multihead_net.parameters(), lr=1e-3)
        self.critic_loss_func = torch.nn.MSELoss()
        
        if len(avg_cstrnt_limit) != num_avg_constraints:
            print(f"{RED}[ERROR] Cstrnts' info is mismatch! Please check the num of cstrnt{RESET}")
            sys.exit()
        # elif num_avg_constraints == 0:
            
        else:
            self.constraint_limits = avg_cstrnt_limit
            self.adaptive_avg_constraints = avg_cstrnt_limit

        self.prob_constraint_limits = 0.2
        self.prob_constraint_threshold = 0.001 # threshold for probablistic constraint
        self.adaptive_prob_constraints = self.prob_constraint_limits
        
        self.alpha = 0.1
        self.t = 20
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, memory): # memory.append([state, action, reward, next_state, mask, [cost1, cost2, cost3, cost4]])
        states, actions, rewards, next_states, masks = [], [], [], [], []
        costs = [[] for _ in range(len(memory[0][5]))]
        prob_costs = []
        
        for m in memory:
            states.append(m[0])
            actions.append(m[1])
            rewards.append(m[2])
            next_states.append(m[3])
            masks.append(m[4])
            for i, cost in enumerate(m[5]):
                costs[i].append(cost)
            prob_costs.append(self.prob_cost_function(m[0], m[3], 1))  # Calculate probabilistic cost for each step

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        masks = torch.tensor(np.array(masks), dtype=torch.float32).to(self.device)
        costs = [torch.tensor(np.array(cost), dtype=torch.float32).to(self.device) for cost in costs]
        prob_costs = torch.tensor(np.array(prob_costs), dtype=torch.float32).to(self.device)  # Convert prob costs to tensor

        values = self.critic_net(states)
        cost_values = self.multihead_net(states)
        returns, advants, cost_advants = self.compute_cost_advantages(rewards, masks, values, costs, cost_values)
        old_mu, old_std = self.actor_net(states)
        pi = self.actor_net.distribution(old_mu, old_std)
        old_log_prob = pi.log_prob(actions).sum(1, keepdim=True)
        
        self.adaptive_avg_constraints = self.adaptive_constraint_thresholding(cost_values, self.constraint_limits)
        self.adaptive_prob_constraint = self.adaptive_prob_constraint_thresholding(prob_costs) ## HERE
        print(f"\n{MAGENTA}prob_cnstrnt: {RESET}{self.prob_constraint_limits}\t{MAGENTA}adaptive_prob_cnstrnt: {RESET}{self.adaptive_prob_constraint}\t{MAGENTA}state[1]: {RESET}{states[:,1]}")

        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n // batch_size):
                b_index = arr[batch_size * i:batch_size * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                mu, std = self.actor_net(b_states)
                
                pi = self.actor_net.distribution(mu, std)
                new_prob = pi.log_prob(b_actions).sum(1, keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                ratio = torch.exp(new_prob - old_prob)

                surrogate_loss = ratio * b_advants
                values = self.critic_net(b_states)
                critic_loss = self.critic_loss_func(values, b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean() # PPO

                cost_values_estimates = self.multihead_net(b_states)
                b_cost_advants = [adv[b_index] for adv in cost_advants]
                actor_loss = self.augmented_objective(actor_loss, cost_values_estimates.t(), b_cost_advants, self.adaptive_avg_constraints, b_advants, old_mu[b_index], old_std[b_index], mu, std)

                ## HERE
                prob_cost_sum = prob_costs.sum()  # Mean of probabilistic costs for the episode
                prob_constraint_loss = self.logarithmic_barrier(prob_cost_sum, self.adaptive_prob_constraint)
                actor_loss += prob_constraint_loss  # Add probabilistic constraint loss

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                cost_value_loss = sum([torch.nn.functional.mse_loss(est.squeeze(), val[b_index]) for est, val in zip(cost_values_estimates.t(), costs)])

                self.multihead_optim.zero_grad()
                cost_value_loss.backward()
                self.multihead_optim.step()
                
    # Get the Kullback - Leibler divergence: Measure of the diff btwn new and old policy:
    # Could be used for the objective function depending on the strategy that needs to be
    def kl_divergence(self, old_mu, old_sigma, mu, sigma):
        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def prob_cost_function(self, s, s_prime, num):
        if (s[num] <= self.prob_constraint_limits).all() and (s_prime[num] <= self.prob_constraint_limits):
            return 0
        else:
            return 1

    # every step에서의 cost의 합이 0.001(0.0572deg)보다 작아야 함
    # Adaptive thresholding for probabilistic constraints
    def adaptive_prob_constraint_thresholding(self, prob_costs):
        current_prob_cost = prob_costs.mean().item()
        adaptive_limit = max(self.prob_constraint_threshold, current_prob_cost + self.alpha * self.prob_constraint_threshold)
        return adaptive_limit

    # Advantage estimation:
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards).to(self.device)
        masks = torch.Tensor(masks).to(self.device)
        # Create an equivalent fullfilled of 0.
        returns = torch.zeros_like(rewards).to(self.device)
        advants = torch.zeros_like(rewards).to(self.device)
        # Init
        running_returns = 0
        previous_value = 0
        running_advants = 0
        # Here we compute A_t the advantage.
        for t in reversed(range(0, len(rewards))):
            # Here we compute the discounted returns. Gamma is the discount factor.
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            #computes the difference between the estimated value at time step t (values.data[t]) and the discounted next value.
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            # Compute advantage
            running_advants = running_tderror + gamma * lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        # Normalization to stabilize final advantage of the history to now.
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants
    
    def compute_cost_advantages(self, rewards, masks, values, cost_rewards, cost_values):
        returns, advantages = self.get_gae(rewards, masks, values)
        cost_advantages = []
        
        for cost_reward, cost_value in zip(cost_rewards, cost_values.t()):
            _, cost_advantage = self.get_gae(cost_reward, masks, cost_value)
            cost_advantages.append(cost_advantage)
        
        return returns, advantages, cost_advantages
    
    def logarithmic_barrier(self, cost, constraint_max):
        indicator = torch.where((cost - constraint_max) <= 0, (cost - constraint_max).clone().detach(), torch.tensor(0, device=cost.device))
        return -torch.log(-indicator)
    
    def augmented_objective(self, actor_loss, cost_values, cost_advants, adaptive_constraints, advants, old_mu, old_sigma, mu, sigma):
        constraint_barrier = sum([self.logarithmic_barrier(cost_advant / (1 - gamma) + cost_value, adaptive_constraint) / self.t 
                                  for cost_value, cost_advant, adaptive_constraint in zip(cost_values, cost_advants, adaptive_constraints)])
        kl_divergence = self.kl_divergence(old_mu, old_sigma, mu, sigma).mean()
        return actor_loss + constraint_barrier.mean() + advants.mean() #+ kl_divergence
    
    def adaptive_constraint_thresholding(self, cost_values, constraint_limits):
        adaptive_limits = []
        for cost_value, constraint_limit in zip(cost_values.t(), constraint_limits):
            current_cost = cost_value.mean().item()
            adaptive_limit = max(constraint_limit, current_cost + self.alpha * constraint_limit)
            adaptive_limits.append(adaptive_limit)
        return adaptive_limits
    
    def save(self):
        # filename = str(filename)
        torch.save(self.actor_optim.state_dict(), self.log_dir + "_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), self.log_dir + "_critic_optimizer")
        torch.save(self.multihead_optim.state_dict(), self.log_dir + "_multihead_optimizer")

    def load(self, log_dir=None):
        # filename = str(filename)
        if log_dir == None:
            log_dir = self.log_dir
        self.actor_optim.load_state_dict(torch.load(log_dir + "_actor_optimizer"))
        self.critic_optim.load_state_dict(torch.load(log_dir + "_critic_optimizer"))
        self.multihead_optim.load_state_dict(torch.load(log_dir + "_multihead_optimizer"))
        

# Creation of a class to normalize the states (Z-score Normalization (Standardization))
class Normalize:
    def __init__(self, N_S, chkpt_dir, train_mode=True, continue_train=False):
        self.mean = np.zeros((N_S,)) # mean
        self.std = np.zeros((N_S, )) # standard
        self.stdd = np.zeros((N_S, )) # variance
        
        self.train_mode = train_mode
        self.continue_train = continue_train

        if not self.continue_train:
            self.n = 0
        else:
            self.n = 1
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_normalize.npy')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        x = np.asarray(x) # x: 들어오는 state 값, n = 0
        if self.train_mode: # 학습하는 모드이면
            self.n += 1 # n += 1
            if self.n == 1: # n = 1이면
                self.mean = x # 현재 state를 평균 mean 값으로 설정
            else:
                old_mean = self.mean.copy() # 현재 mean을 old_mean으로 설정
                self.mean = old_mean + (x - old_mean) / self.n # 다시 mean 구하기
                self.stdd = self.stdd + (x - old_mean) * (x - self.mean) #  다시 std 구하기
            if self.n > 1:
                self.std = np.sqrt(self.stdd / (self.n - 1))
            else:
                self.std = self.mean

        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x
    
    def update(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0) + 1e-8
    
    def save_params(self):
        np.save(self.checkpoint_file, {'mean': self.mean, 'std': self.std, 'stdd': self.stdd})

    def load_params(self, load_model_dir=None):
        if load_model_dir is None:
            load_model_dir = self.checkpoint_file
        params = np.load(load_model_dir, allow_pickle=True).item()
        self.mean = params['mean']
        self.std = params['std']
        self.stdd = params['stdd']

def main():
    env = gym.make('Walker2d-v4', render_mode='rgb_array')

    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]

    average_constraint_limit = []
    average_constraint_limit.append(0.2) # y angle of the torso
    average_constraint_limit.append(0.5)  # x vel of the torso
    average_constraint_limit.append(1)  # angle of the leg joint
    average_constraint_limit.append(1)  # angle of the left leg joint

    # ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=4, avg_cstrnt_limit=average_constraint_limit)
    ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=1, avg_cstrnt_limit=[0.5])
    normalize = Normalize(N_S, log_dir, train_mode=True)
    
    # ppo.actor_net.load_model('../runs/20240725_19-01-40/_actor')
    # ppo.actor_net.eval()
    # ppo.critic_net.load_model('../runs/20240725_19-01-40/_critic')
    # ppo.critic_net.eval()
    # ppo.load('../runs/20240725_19-01-40/')
    # normalize.load_params('../runs/20240725_19-01-40/_normalize.npy')

    episodes = 0
    episode_data = []
    constraint_data = []

    prob_cstrnt1 = []
    prob_cstrnt1_next = []

    for iter in tqdm(range(Iter)):
        memory = deque()
        scores = []
        steps = 0
        avg_cstrnt1 = []
        avg_cstrnt2 = []
        avg_cstrnt3 = []
        avg_cstrnt4 = []
        while steps < 2048: #Horizon
            episodes += 1
            state, _ = env.reset()
            state = normalize(state)
            score = 0
            for _ in range(MAX_STEP):
                steps += 1
                
                action = ppo.actor_net.choose_action(state)
                next_state, reward, truncated, terminated, info = env.step(action)
                next_state = normalize(next_state)
                done = truncated or terminated
                
                cost1 = next_state[1] # y angle of the torso
                cost2 = next_state[8] # x velocity of the torso
                cost3 = -next_state[3]
                cost4 = -next_state[6]
                prob1 = state[1]
                prob1_next = next_state[1]

                mask = (1-done)*1
                memory.append([state, action, reward, next_state, mask, [cost1, cost2, cost3, cost4]])
                avg_cstrnt1.append(state[1])
                avg_cstrnt2.append(state[8])
                avg_cstrnt3.append(-state[3])
                avg_cstrnt4.append(-state[6])
                prob_cstrnt1.append(prob1)
                prob_cstrnt1_next.append(prob1_next)
                score += reward
                state = next_state

                if done:
                    break
            
            scores.append(score)
        
        score_avg = np.mean(scores)
        cstrnt1_avg = np.mean(avg_cstrnt1)
        cstrnt2_avg = np.mean(avg_cstrnt2)
        cstrnt3_avg = np.mean(avg_cstrnt3)
        cstrnt4_avg = np.mean(avg_cstrnt4)
        
        # if (cstrnt1_avg <= ppo.adaptive_avg_constraints[0]) & (cstrnt2_avg <= ppo.adaptive_avg_constraints[1]):
        #     print(f"\n{episodes} episode score is {score_avg:.2f}, cstrnt1 is {GREEN}{cstrnt1_avg:.3f}/ {ppo.adaptive_avg_constraints[0]:.3f}{RESET}, cstrnt2 is {GREEN}{cstrnt2_avg:.3f}/ {ppo.adaptive_avg_constraints[1]:.3f}{RESET}")
        # elif (cstrnt1_avg <= ppo.adaptive_avg_constraints[0]) & (cstrnt2_avg > ppo.adaptive_avg_constraints[1]):
        #     print(f"\n{episodes} episode score is {score_avg:.2f}, cstrnt1 is {GREEN}{cstrnt1_avg:.3f}/ {ppo.adaptive_avg_constraints[0]:.3f}{RESET}, cstrnt2 is {RED}{cstrnt2_avg:.3f}/ {ppo.adaptive_avg_constraints[1]:.3f}{RESET}")
        # elif (cstrnt1_avg > ppo.adaptive_avg_constraints[0]) & (cstrnt2_avg <= ppo.adaptive_avg_constraints[1]):
        #     print(f"\n{episodes} episode score is {score_avg:.2f}, cstrnt1 is {RED}{cstrnt1_avg:.3f}/ {ppo.adaptive_avg_constraints[0]:.3f}{RESET}, cstrnt2 is {GREEN}{cstrnt2_avg:.3f}/ {ppo.adaptive_avg_constraints[1]:.3f}{RESET}")
        # else:
        #     print(f"\n{episodes} episode score is {score_avg:.2f}, cstrnt1 is {RED}{cstrnt1_avg:.3f}/ {ppo.adaptive_avg_constraints[0]:.3f}{RESET}, cstrnt2 is {RED}{cstrnt2_avg:.3f}/ {ppo.adaptive_avg_constraints[1]:.3f}{RESET}")
        #
        # if (cstrnt3_avg <= ppo.adaptive_avg_constraints[0]) & (cstrnt4_avg <= ppo.adaptive_avg_constraints[1]):
        #     print(f"\n\t\t\t\tcstrnt3 is {GREEN}{cstrnt3_avg:.3f}/ {ppo.adaptive_avg_constraints[2]:.3f}{RESET}, cstrnt4 is {GREEN}{cstrnt4_avg:.3f}/ {ppo.adaptive_avg_constraints[3]:.3f}{RESET}")
        # elif (cstrnt3_avg <= ppo.adaptive_avg_constraints[0]) & (cstrnt4_avg > ppo.adaptive_avg_constraints[1]):
        #     print(f"\n\t\t\tcstrnt3 is {GREEN}{cstrnt3_avg:.3f}/ {ppo.adaptive_avg_constraints[2]:.3f}{RESET}, cstrnt4 is {RED}{cstrnt4_avg:.3f}/ {ppo.adaptive_avg_constraints[3]:.3f}{RESET}")
        # elif (cstrnt3_avg > ppo.adaptive_avg_constraints[0]) & (cstrnt4_avg <= ppo.adaptive_avg_constraints[1]):
        #     print(f"\n\t\t\tcstrnt3 is {RED}{cstrnt3_avg:.3f}/ {ppo.adaptive_avg_constraints[2]:.3f}{RESET}, cstrnt4 is {GREEN}{cstrnt4_avg:.3f}/ {ppo.adaptive_avg_constraints[3]:.3f}{RESET}")
        # else:
        #     print(f"\n\t\t\tcstrnt3 is {RED}{cstrnt3_avg:.3f}/ {ppo.adaptive_avg_constraints[2]:.3f}{RESET}, cstrnt4 is {RED}{cstrnt4_avg:.3f}/ {ppo.adaptive_avg_constraints[3]:.3f}{RESET}")
        
        episode_data.append([iter + 1, score_avg])
        constraint_data.append([iter + 1, ppo.adaptive_avg_constraints[0], cstrnt1_avg, ppo.adaptive_avg_constraints[0], cstrnt2_avg,
                                ppo.adaptive_avg_constraints[0], cstrnt3_avg, ppo.adaptive_avg_constraints[0], cstrnt4_avg])
        
        if (iter + 1) % save_freq == 0:
            save_flag = True

            if save_flag:
                ppo.actor_net.save_model()
                ppo.critic_net.save_model()
                ppo.multihead_net.save_model()
                ppo.save()
                normalize.save_params()
                print(f"\n{GREEN} >> Successfully saved models! {RESET}")

                np.save(log_dir + "reward.npy", episode_data)
                np.save(log_dir + "constraint.npy", constraint_data)
                np.save(log_dir + "prob_constraint.npy", prob_cstrnt1)
                np.save(log_dir + "prob_constraint_next.npy", prob_cstrnt1_next)
                print(f"{GREEN} >> Successfully saved reward & constraint data! {RESET}")
                
                save_flag = False

        ppo.train(memory)
        

if __name__ == "__main__":
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    log_dir = f"../runs/{current_time}/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"{YELLOW}[MODEL/TENSORBOARD]{RESET} The data will be saved in {YELLOW}{log_dir}{RESET} directory!")

    # tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', f"../runs/franka_cabinet/{current_time}", '--port', '6300'])
    # url = tb.launch()
    # webbrowser.open_new(url)
    
    main()