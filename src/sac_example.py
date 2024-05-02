# import os
# import sys
# import time

# import gymnasium as gym

# import math
# import numpy as np

# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim # Adam, Cross-Entropy, etc.
# import torch.nn.functional as F # relu, softmax, etc.
# from torch.distributions.normal import Normal
# import matplotlib.pyplot as plt

# import pandas as pd
# import seaborn as sns

# class ReplayBuffer():
#     def __init__(self, max_size, input_shape, n_actions):
#         self.mem_size = max_size
#         self.mem_cntr = 0
#         self.state_memory = np.zeros((self.mem_size, *input_shape))
#         self.new_state_memory = np.zeros((self.mem_size, *input_shape))
#         self.action_memory = np.zeros((self.mem_size, n_actions))
#         self.reward_memory = np.zeros(self.mem_size)
#         self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

#     def store_transition(self, state, action, reward, state_, done):
#         index = self.mem_cntr % self.mem_size

#         self.state_memory[index] = state
#         self.new_state_memory[index] = state_
#         self.action_memory[index] = action
#         self.reward_memory[index] = reward
#         self.terminal_memory[index] = done

#         self.mem_cntr += 1

#     def sample_buffer(self, batch_size):
#         max_mem = min(self.mem_cntr, self.mem_size)

#         batch = np.random.choice(max_mem, batch_size)

#         states = self.state_memory[batch]
#         states_ = self.new_state_memory[batch]
#         actions = self.action_memory[batch]
#         rewards = self.reward_memory[batch]
#         dones = self.terminal_memory[batch]

#         return states, actions, rewards, states_, dones
    
# class CriticNetwork(nn.Module):
#     def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
#             name='critic', chkpt_dir='tmp/sac'):
#         super(CriticNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

#         self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.q = nn.Linear(self.fc2_dims, 1)

#         self.optimizer = optim.Adam(self.parameters(), lr=beta)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         self.to(self.device)

#     def forward(self, state, action):
#         action_value = self.fc1(torch.cat([state, action], dim=1))
#         action_value = F.relu(action_value)
#         action_value = self.fc2(action_value)
#         action_value = F.relu(action_value)

#         q = self.q(action_value)

#         return q

#     def save_checkpoint(self):
#         torch.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(torch.load(self.checkpoint_file))

# class ValueNetwork(nn.Module):
#     def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
#             name='value', chkpt_dir='tmp/sac'):
#         super(ValueNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

#         self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
#         self.v = nn.Linear(self.fc2_dims, 1)

#         self.optimizer = optim.Adam(self.parameters(), lr=beta)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         self.to(self.device)

#     def forward(self, state):
#         state_value = self.fc1(state)
#         state_value = F.relu(state_value)
#         state_value = self.fc2(state_value)
#         state_value = F.relu(state_value)

#         v = self.v(state_value)

#         return v

#     def save_checkpoint(self):
#         torch.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(torch.load(self.checkpoint_file))

# class ActorNetwork(nn.Module):
#     def __init__(self, alpha, input_dims, max_action, fc1_dims=256, 
#             fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
#         super(ActorNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
#         self.max_action = max_action
#         self.reparam_noise = 1e-6

#         self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.mu = nn.Linear(self.fc2_dims, self.n_actions)
#         self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         self.to(self.device)

#     def forward(self, state):
#         prob = self.fc1(state)
#         prob = F.relu(prob)
#         prob = self.fc2(prob)
#         prob = F.relu(prob)

#         mu = self.mu(prob)
#         sigma = self.sigma(prob)

#         mu = torch.tanh(self.mu(prob)) * torch.tensor(self.max_action).to(self.device)
#         sigma = F.softplus(self.sigma(prob))

#         return mu, sigma

#     def sample_normal(self, state, reparameterize=True):
#         mu, sigma = self.forward(state)
#         probabilities = Normal(mu, sigma)

#         if reparameterize:
#             actions = probabilities.rsample()
#         else:
#             actions = probabilities.sample()
#         action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
#         action = action[:, :1]
#         log_probs = probabilities.log_prob(actions)
#         log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
#         log_probs = log_probs.sum(1, keepdim=True)

#         return action, log_probs

#     def save_checkpoint(self):
#         torch.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(torch.load(self.checkpoint_file))


# class Agent():
#     def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
#             env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
#             layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
#         self.gamma = gamma
#         self.tau = tau
#         self.memory = ReplayBuffer(max_size, input_dims, n_actions)
#         self.batch_size = batch_size
#         self.n_actions = n_actions

#         self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
#                     name='actor', max_action=env.action_space.high)
#         self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
#                     name='critic_1')
#         self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
#                     name='critic_2')
#         self.value = ValueNetwork(beta, input_dims, name='value')
#         self.target_value = ValueNetwork(beta, input_dims, name='target_value')

#         self.scale = reward_scale
#         self.update_network_parameters(tau=1)

#     def choose_action(self, observation):
#         state = torch.Tensor(np.array([observation]).astype(np.float32)).to(self.actor.device)
#         actions, _ = self.actor.sample_normal(state, reparameterize=False)
#         ## state는 잘 나오고 있음, sample_normal 부분 확인해보면 될 것 같음!
#         return actions.cpu().detach().numpy().astype(np.float32)[0]

#     def remember(self, state, action, reward, new_state, done):
#         self.memory.store_transition(state, action, reward, new_state, done)

#     def update_network_parameters(self, tau=None):
#         if tau is None:
#             tau = self.tau

#         target_value_params = self.target_value.named_parameters()
#         value_params = self.value.named_parameters()

#         target_value_state_dict = dict(target_value_params)
#         value_state_dict = dict(value_params)

#         for name in value_state_dict:
#             value_state_dict[name] = tau*value_state_dict[name].clone() + \
#                     (1-tau)*target_value_state_dict[name].clone()

#         self.target_value.load_state_dict(value_state_dict)

#     def save_models(self):
#         print('.... saving models ....')
#         self.actor.save_checkpoint()
#         self.value.save_checkpoint()
#         self.target_value.save_checkpoint()
#         self.critic_1.save_checkpoint()
#         self.critic_2.save_checkpoint()

#     def load_models(self):
#         print('.... loading models ....')
#         self.actor.load_checkpoint()
#         self.value.load_checkpoint()
#         self.target_value.load_checkpoint()
#         self.critic_1.load_checkpoint()
#         self.critic_2.load_checkpoint()

#     def learn(self):
#         if self.memory.mem_cntr < self.batch_size:
#             return

#         state, action, reward, new_state, done = \
#                 self.memory.sample_buffer(self.batch_size)

#         reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
#         done = torch.tensor(done).to(self.actor.device)
#         state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
#         state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
#         action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

#         value = self.value(state).view(-1)
#         value_ = self.target_value(state_).view(-1)
#         value_[done] = 0.0

#         actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
#         log_probs = log_probs.view(-1)
#         q1_new_policy = self.critic_1.forward(state, actions)
#         q2_new_policy = self.critic_2.forward(state, actions)
#         critic_value = torch.min(q1_new_policy, q2_new_policy)
#         critic_value = critic_value.view(-1)

#         self.value.optimizer.zero_grad()
#         value_target = critic_value - log_probs
#         value_loss = 0.5 * F.mse_loss(value, value_target)
#         value_loss.backward(retain_graph=True)
#         self.value.optimizer.step()

#         actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
#         log_probs = log_probs.view(-1)
#         q1_new_policy = self.critic_1.forward(state, actions)
#         q2_new_policy = self.critic_2.forward(state, actions)
#         critic_value = torch.min(q1_new_policy, q2_new_policy)
#         critic_value = critic_value.view(-1)
        
#         actor_loss = log_probs - critic_value
#         actor_loss = torch.mean(actor_loss)
#         self.actor.optimizer.zero_grad()
#         actor_loss.backward(retain_graph=True)
#         self.actor.optimizer.step()

#         self.critic_1.optimizer.zero_grad()
#         self.critic_2.optimizer.zero_grad()
#         q_hat = self.scale*reward + self.gamma*value_
#         q1_old_policy = self.critic_1.forward(state, action).view(-1)
#         q2_old_policy = self.critic_2.forward(state, action).view(-1)
#         critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
#         critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

#         critic_loss = critic_1_loss + critic_2_loss
#         critic_loss.backward()
#         self.critic_1.optimizer.step()
#         self.critic_2.optimizer.step()

#         self.update_network_parameters()

# episode_durations = []
# # episode durations 리스트 형성

# # Create and wrap the environment
# env = gym.make("InvertedPendulum-v4", render_mode="human") # ADD human
# wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

# total_num_episodes = int(5e3)  # Total number of episodes
# # Observation-space of InvertedPendulum-v4 (4)
# obs_space_dims = env.observation_space.shape[0]
# # Action-space of InvertedPendulum-v4 (1)
# action_space_dims = env.action_space.shape[0]
# rewards_over_seeds = []

# for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
#     # set seed
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     # observation, info=wrapped_env.reset(seed=seed) # ADD
#     # wrapped_env.render() # ADD

#     # Reinitialize agent every seed
#     agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=2)
#     reward_over_episodes = []

#     for episode in range(total_num_episodes):
#         # gymnasium v26 requires users to set seed while resetting the environment
#         obs, info = wrapped_env.reset(seed=seed)
#         episode_reward = 0

#         done = False
#         while not done:
#             action = agent.choose_action(obs)
#             print(action)

#             # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
#             # These represent the next observation, the reward from the step,
#             # if the episode is terminated, if the episode is truncated and
#             # additional info from the step
#             new_obs, reward, terminated, truncated, info = wrapped_env.step(action)

#             # End the episode when either truncated or terminated is true
#             #  - truncated: The episode duration reaches max number of timesteps
#             #  - terminated: Any of the state space values is no longer finite.
#             done = terminated or truncated

#             agent.remember(obs, action, reward, new_obs, done)
#             agent.learn()

#             observation = new_obs
#             episode_reward += reward

#             if done:
#                 break  

#         reward_over_episodes.append(wrapped_env.return_queue[-1])
#         agent.save_models()

#         if episode % 1000 == 0:
#             avg_reward = int(np.mean(wrapped_env.return_queue))
#             print("Episode:", episode, "Average Reward:", avg_reward)

#     rewards_over_seeds.append(reward_over_episodes)


# # %%
# # Plot learning curve
# # ~~~~~~~~~~~~~~~~~~~
# #

# rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
# df1 = pd.DataFrame(rewards_to_plot).melt()
# df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
# sns.set(style="darkgrid", context="talk", palette="rainbow")
# sns.lineplot(x="episodes", y="reward", data=df1).set(
#     title="REINFORCE for InvertedPendulum-v4"
# )
# plt.show()



########################################################################################
########################################################################################

'''
Soft Actor-Critic version 1
using state value function: 1 V net, 1 target V net, 2 Q net, 1 policy net
paper: https://arxiv.org/pdf/1801.01290.pdf
'''

import os
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import display
from reacher import Reacher

import argparse
import time


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, activation=F.relu, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation
        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.activation = activation
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, activation=F.relu, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = 10.
        self.num_actions = num_actions
        self.activation = activation

        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        deterministic evaluation provides better performance according to the original paper;
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        ''' stochastic evaluation '''
        log_prob = Normal(mean, std).log_prob(mean + std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        ''' deterministic evaluation '''
        # log_prob = Normal(mean, std).log_prob(mean) - torch.log(1. - torch.tanh(mean).pow(2) + epsilon) -  np.log(self.action_range)
        '''
         both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
         the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
         needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
         '''
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.tanh(mean + std*z)        
        action = torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return (self.action_range*a).numpy()


def update(batch_size, reward_scale, gamma=0.99,soft_tau=1e-2):
    alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    # print('sample:', state, action,  reward, done)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value    = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

    reward = reward_scale*(reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std

# Training Q Function
    target_value = target_value_net(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value # if done==1, only reward
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())


    soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()  

# Training Value Function
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action),soft_q_net2(state, new_action))
    target_value_func = predicted_new_q_value - alpha * log_prob # for stochastic training, it equals to expectation over action
    value_loss = value_criterion(predicted_value, target_value_func.detach())

    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

# Training Policy Function
    ''' implementation 1 '''
    policy_loss = (alpha * log_prob - predicted_new_q_value).mean()
    ''' implementation 2 '''
    # policy_loss = (alpha * log_prob - soft_q_net1(state, new_action)).mean()  # Openai Spinning Up implementation
    ''' implementation 3 '''
    # policy_loss = (alpha * log_prob - (predicted_new_q_value - predicted_value.detach())).mean() # max Advantage instead of Q to prevent the Q-value drifted high

    ''' implementation 4 '''  # version of github/higgsfield
    # log_prob_target=predicted_new_q_value - predicted_value
    # policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    # mean_lambda=1e-3
    # std_lambda=1e-3
    # mean_loss = mean_lambda * mean.pow(2).mean()
    # std_loss = std_lambda * log_std.pow(2).mean()
    # policy_loss += mean_loss + std_loss


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # print('value_loss: ', value_loss)
    # print('q loss: ', q_value_loss1, q_value_loss2)
    # print('policy loss: ', policy_loss )


# Soft update the target value net
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(  # copy data value into target parameters
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    return predicted_new_q_value.mean()


def plot(rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac.png')
    # plt.show()


DETERMINISTIC=False

# choose env
ENV = ['Pendulum', 'Reacher'][1] # 0: Pendulum, 1: Reacher
if ENV == 'Reacher':
    # intialization
    # NUM_JOINTS=4
    # LINK_LENGTH=[200, 140, 80, 50]
    # INI_JOING_ANGLES=[0.1, 0.1, 0.1, 0.1]
    NUM_JOINTS=2
    LINK_LENGTH=[200, 140]
    INI_JOING_ANGLES=[0.1, 0.1]
    SCREEN_SIZE=1000
    SPARSE_REWARD=False
    SCREEN_SHOT=False
    env=Reacher(screen_size=SCREEN_SIZE, num_joints=NUM_JOINTS, link_lengths = LINK_LENGTH, \
    ini_joint_angles=INI_JOING_ANGLES, target_pos = [369,430], render=True,  change_goal=False)
    action_dim = env.num_actions
    state_dim  = env.num_observations
elif ENV == 'Pendulum':
    env = NormalizedActions(gym.make("Pendulum-v0"))
    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]

hidden_dim = 512

value_net        = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim, activation=F.relu).to(device)

soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, activation=F.relu).to(device)

print('(Target) Value Network: ', value_net)
print('Soft Q Network (1,2): ', soft_q_net1)
print('Policy Network: ', policy_net)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = int(1e6)
replay_buffer = ReplayBuffer(replay_buffer_size)


# hyper-parameters
max_episodes  = 1000
max_steps   = 20 if ENV ==  'Reacher' else 150  # Pendulum needs 150 steps per episode to learn well, cannot handle 20
frame_idx   = 0
batch_size  = 128
explore_steps = 0
rewards     = []
reward_scale=10.0
model_path = 'model/sac/sac.pth'


if __name__ == '__main__':
    if args.train:
        # training loop
        for eps in range(max_episodes):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()

            episode_reward = 0
            
            
            for step in range(max_steps):
                if frame_idx >= explore_steps:
                    action = policy_net.get_action(state, deterministic=DETERMINISTIC)
                else:
                    action = policy_net.sample_action()
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action)
                    env.render()

                replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                frame_idx += 1
                
                if len(replay_buffer) > batch_size:
                    _=update(batch_size, reward_scale)
                
                if done:
                    break

            if eps % 20 == 0 and eps>0:
                plot(rewards)
                torch.save(policy_net.state_dict(), os.path.join(model_path))

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)
            rewards.append(episode_reward)
        torch.save(policy_net.state_dict(), os.path.join(model_path)) 


    if args.test:
        policy_net.load_state_dict(torch.load(os.path.join(model_path)))
        policy_net.eval()
        for eps in range(10):
            if ENV == 'Reacher':
                state = env.reset(SCREEN_SHOT)
            elif ENV == 'Pendulum':
                state =  env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = policy_net.get_action(state, deterministic = DETERMINISTIC)
                if ENV ==  'Reacher':
                    next_state, reward, done, _ = env.step(action, SPARSE_REWARD, SCREEN_SHOT)
                elif ENV ==  'Pendulum':
                    next_state, reward, done, _ = env.step(action)
                    env.render() 

                episode_reward += reward
                state=next_state

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)