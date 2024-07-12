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

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_time = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
log_dir = f"../runs/{current_time}/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', f"../runs/franka_cabinet/{current_time}", '--port', '6300'])
# url = tb.launch()
# webbrowser.open_new(url)

#learning rate backward propagation NN action
lr_actor = 0.0003
#learning rate backward propagation NN state value estimation
lr_critic = 0.0003
#Number of Learning Iteration we want to perform
Iter = 100000
#Number max of step to realise in one episode. 
MAX_STEP = 1000
#How rewards are discounted.
gamma =0.98
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

    def forward(self,s):
      # Use of tanh activation function is recommanded : bounded [-1,1],
      # gives some non-linearity, and tends to give some stability.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        # mu action output of the NN.
        mu = self.mu(x)
        #log_sigma action output of the NN
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,s):
      # Choose action in the continuous action space using normal distribution
      # defined by mu and sigma of each actions returned by the NN.
        s = torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0).to(self.device)
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().cpu().numpy().squeeze(0)
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    

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

    def forward(self,s):
      # Use of tanh activation function is recommanded.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    
class PPO:
    def __init__(self, N_S, N_A, log_dir):
        self.log_dir = log_dir
        
        self.actor_net = Actor(N_S, N_A, log_dir)
        self.critic_net = Critic(N_S, log_dir)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=1e-3, weight_decay=1e-3)
        self.critic_loss_func = torch.nn.MSELoss()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, memory):
        states, actions, rewards, masks = [], [], [], []
        
        for m in memory:
            states.append(m[0])
            actions.append(m[1])
            rewards.append(m[2])
            masks.append(m[3])
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        masks = torch.tensor(np.array(masks), dtype=torch.float32).to(self.device)

        # Use critic network defined in Model.py
        # This function enables to get the current state value V(S).
        values = self.critic_net(states)
        # Get advantage.
        returns,advants = self.get_gae(rewards,masks,values)
        #Get old mu and std.
        old_mu,old_std = self.actor_net(states)
        #Get the old distribution.
        pi = self.actor_net.distribution(old_mu,old_std)
        #Compute old policy.
        old_log_prob = pi.log_prob(actions).sum(1,keepdim=True)

        # Everything happens here
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n//batch_size):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                #New parameter of the policy distribution by action.
                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                #Regularisation fixed KL : does not work as good as following clipping strategy
                # empirically.
                # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants
                values = self.critic_net(b_states)
                # MSE Loss : (State action value - State value)^2
                critic_loss = self.critic_loss_func(values,b_returns)
                # critic_loss = critic_loss - beta*KL_penalty

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                #Clipping strategy
                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)
                clipped_loss =ratio*b_advants
                # Actual loss
                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                
                # walker_xvel = torch.tensor([get_walker_x_velocity(state) for state in b_states], dtype=torch.float32).to(self.device)
                # actor_loss = augmented_objective(actor_loss, walker_xvel, 3, 20)
                
                #Now that we have the loss, we can do the backward propagation to learn : everything is here.
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
    # Get the Kullback - Leibler divergence: Measure of the diff btwn new and old policy:
    # Could be used for the objective function depending on the strategy that needs to be
    # teste.
    def kl_divergence(self,old_mu,old_sigma,mu,sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    # Advantage estimation:
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards).to(self.device)
        masks = torch.Tensor(masks).to(self.device)
        #Create an equivalent fullfilled of 0.
        returns = torch.zeros_like(rewards).to(self.device)
        advants = torch.zeros_like(rewards).to(self.device)
        #Init
        running_returns = 0
        previous_value = 0
        running_advants = 0
        #Here we compute A_t the advantage.
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
        #Normalization to stabilize final advantage of the history to now.
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

    def save(self, filename):
        filename = str(filename)
        torch.save(self.actor_net.state_dict(), filename + "_actor")
        torch.save(self.critic_net.state_dict(), filename + "_critic")
        torch.save(self.actor_optim.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        filename = str(filename)
        self.actor_net.load_state_dict(torch.load(filename + "_actor"))
        self.critic_net.load_state_dict(torch.load(filename + "_critic"))
        self.actor_optim.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer"))
        

# Creation of a class to normalize the states
class Normalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
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
    
    def save_params(self, path):
        np.save(path, {'mean': self.mean, 'std': self.std})

    def load_params(self, path):
        params = np.load(path, allow_pickle=True).item()
        self.mean = params['mean']
        self.std = params['std']
    
def test_model(env, model, episodes=10):
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = model.actor_net.choose_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        scores.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    print(f"Average Reward over {episodes} episodes: {np.mean(scores)}")
    env.close()
    
def get_walker_x_velocity(state):
    x_vel = state[8]
    return x_vel

def logarithmic_barrier(state, constraint_max):
    return -torch.log(-(state - constraint_max))

def augmented_objective(actor_loss, state, constraint_max, t):
    constraint_barrier = logarithmic_barrier(state, constraint_max) / t
    return actor_loss + constraint_barrier.mean()

def main():
    env = gym.make('Walker2d-v4', render_mode='rgb_array')

    #Number of state and action
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]

    # Random seed initialization
    # env.seed(500)
    # torch.manual_seed(500)
    # np.random.seed(500)

    # Run the Ppo class
    frames = []
    ppo = PPO(N_S, N_A, log_dir)
    # ppo.actor_net.load_model("../runs/20240708_11-19-08/ppo/100000/")
    # ppo.critic_net.load_model("../runs/20240708_11-19-08/ppo/100000/")
    
    # Normalisation for stability, fast convergence... always good to do.
    normalize = Normalize(N_S)
    episodes = 0
    eva_episodes = 0
    episode_data = []
    state, _ = env.reset()

    for iter in tqdm(range(Iter)):
        memory = deque()
        scores = []
        steps = 0
        xvel = []
        while steps < 2048: #Horizon
            episodes += 1
            state, _ = env.reset()
            s = normalize(state)
            score = 0
            for _ in range(MAX_STEP):
                steps += 1
                #Choose an action: detailed in PPO.py
                # The action is a numpy array of 17 elements. It means that in the 17 possible directions of action we have a specific value in the continuous space.
                # Exemple : the first coordinate correspond to the Torque applied on the hinge in the y-coordinate of the abdomen: this is continuous space.
                a = ppo.actor_net.choose_action(s)
                # print(f"{YELLOW}walker velocity: {RESET}", s[8]) # 3
                #Environnement reaction to the action : There is a reaction in the 376 elements that characterize the space :
                # Exemple : the first coordinate of the states is the z-coordinate of the torso (centre) and using env.step(a), we get the reaction of this state and
                # of all the other ones after the action has been made.
                s_ , r ,truncated, terminated ,info = env.step(a)
                s_ = normalize(s_)
                done = truncated or terminated

                # Do we continue or do we terminate an episode?
                mask = (1-done)*1
                memory.append([s,a,r,mask])
                # print('s: ', s)
                # print('a: ', a)
                # print('r: ', r)
                # print('mask: ', mask)
                xvel.append(s[8])
                score += r
                s = s_

                if done:
                    break
            # with open('log_' + args.env_name  + '.txt', 'a') as outfile:
            #     outfile.write('\t' + str(episodes)  + '\t' + str(score) + '\n')
            scores.append(score)
        score_avg = np.mean(scores)
        xvel_avg = np.mean(xvel)
        print('{} episode score is {:.2f}, average_xvel is {:.3f}'.format(episodes, score_avg, xvel_avg))
        episode_data.append([iter + 1, score_avg])
        if (iter + 1) % save_freq == 0:
            save_flag = True

            if save_flag:
                ppo.actor_net.save_model()
                ppo.critic_net.save_model()
                print(f"{GREEN} >> Successfully saved models! {RESET}")
                # path = log_dir + "ppo/" + str((iter + 1)) + "/"
                # os.makedirs(path, exist_ok=True)
                # if not os.path.exists(path):
                #     os.makedirs(path)
                # ppo.save(path)

                np.save(log_dir + "reward.npy", episode_data)
                save_flag = False

        ppo.train(memory)
        

if __name__ == "__main__":
    main()