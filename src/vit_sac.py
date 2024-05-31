import os
import time
import asyncio

import gymnasium as gym

import mujoco
import mujoco.viewer

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim # Adam, Cross-Entropy, etc.
import torch.nn.functional as F # relu, softmax, etc.
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

import mjc_pybind.mjc_controller as mjctrl
from transformers import ViTModel, AutoFeatureExtractor


# define
M_PI = math.pi
M_PI_2 = M_PI/2
M_PI_4 = M_PI/4

DOF = 9

MAX_EPISODES = 1000
MAX_STEPS = 1000 # 한 번 학습할 때 몇 번 iteration 돌릴 건지

torch.set_default_dtype(torch.float32)

# Class: ViTFeatureExtraction
class ViTFeatureExtraction:
    def __init__(self, model_name='google/vit-base-patch16-224'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # print(self.model.eval())

    def get_feature(self, img):
        inputs = self.feature_extractor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        outputs = outputs.last_hidden_state.squeeze(0)
        outputs = torch.flatten(outputs)
        
        return outputs.detach().cpu().numpy()
    
# vit = ViTFeatureExtraction()
# from PIL import Image
# img_path = './img/reacher.png'
# img = Image.open(img_path).convert('RGB')
# rgb_array = np.array(img)

# print("=====================\n", vit.get_feature(rgb_array).shape)

# Class: ReplayBuffer
class ReplayBuffer():
    '''
        ReplayBuffer:
        - 재현 버퍼, 재생 버퍼/메모리
        - 경험에 대한 기록을 저장
        - 환경에서 정책을 실행할 대 경험의 궤적을 저장
    '''
    def __init__(self, max_size, input_shape, n_actions):
        # self
        self.mem_size = max_size # mem_size: memory size
        self.mem_cntr = 0 # mem_cntr: memory counter
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        # transition: state들간에 이동하는 것을 의미
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# Class: CriticNetwork
class CriticNetwork(nn.Module):
    '''
        CriticNetwork:
        - 주어진 state에서의 value function을 학습
        - agent가 선택한 행동의 value를 평가, 이를 통해 value를 update
    '''
    def __init__(self, learning_rate, input_dims, n_actions, fc1_dims=256, fc2_dims=256, 
                 name='critic', chkpt_dir='tmp/sac'):
        # initialize
        super(CriticNetwork, self).__init__()

        # self
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # action value를 평가하는 부분
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


# Class: ValueNetwork
class ValueNetwork(nn.Module):
    '''
        ValueNetwork:
        - 
    '''
    def __init__(self, learning_rate, input_dims, fc1_dims=256, fc2_dims=256, 
                 name='value', chkpt_dir='tmp/sac'):
        # initialize
        super(ValueNetwork, self).__init__()

        # self
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # state value를 평가하는 부분
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# Class: ActorNetwork
class ActorNetwork(nn.Module):
    '''
        ActorNetwork:
        - policy network이라고도 함
        - 주어진 state에서 각 가능한 action에 대한 확률 분포를 출력하는 역할을 함
        - actor는 주어진 상태에서 적절한 행동을 선택하는 역할을 하며, 이를 통해 정책을 결정
    '''
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=3, name='actor', chkpt_dir='tmp/sac'):
        # initialize
        super(ActorNetwork, self).__init__()

        # self
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions # present: 2
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action # 가능한 action 값의 최대 크기를 나타냄
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        '''
            mu: state를 입력으로 받아서 각 가능한 action에 대한 평균(mean)을 계산
                -> 각 action의 평균을 나타냄, 확률 분포의 형태로 출력됨
            sigma: state를 입력으로 받아서 각 가능한 action에 대한 표준편차(variance)을 계산
                -> 각 action의 표준편차을 나타냄, 확률 분포의 형태로 출력됨

            => mu와 sigma를 통해 policy network가 출력하는 확률분포를 나타내는 역할을 함
            => 이 확률 분포를 통해 agent는 주어진 state에서 action을 sampling하여 env와 상호작용하게 됨
        '''
        self.mu = nn.Linear(fc2_dims, max_action.shape[0])
        self.sigma = nn.Linear(fc2_dims, max_action.shape[0])

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        # mu = self.mu(prob)
        # sigma = self.sigma(prob)

        # sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        # policy network에서 출력한 raw action 값을 조정해서 최종 action을 결정하는 부분
        mu = torch.tanh(self.mu(prob)) * torch.tensor(self.max_action).to(self.device)
        sigma = F.softplus(self.sigma(prob)) # 출력된 표준편차값에 대해서 softplus 함수를 적용해서 양수를 보장, 표준편차값 조절

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state) # mu와 sigma를 만들어내고, 가공하는 부분
        probabilities = Normal(mu, sigma) # 확률 분포를 만드는 부분

        if reparameterize: # True인 경우
            actions = probabilities.rsample() # re-parameter화 트릭을 적용한 sampling
        else:
            actions = probabilities.sample() # 일반적인 sampling

        # sampling된 action을 [-1,1] 범위로 조정하고, 가능한 행동값의 최대 크기로 scaling함
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions) # sampling된 action에 대한 log 확률을 계산
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise) # log 확률에 따라 re-parameter화 트릭에 의한 보정 항을 더해줌
        log_probs = log_probs.sum(1, keepdim=True) # 각 action의 log 확률을 합산하여 반환

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class SACAgent():
    def __init__(self, alpha=0.0003, learning_rate=0.0003, input_dims=[2],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        # self.max_action = env.action_space.high
        print("===================================", input_dims)
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(learning_rate, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(learning_rate, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(learning_rate, input_dims, name='value')
        self.target_value = ValueNetwork(learning_rate, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.Tensor(np.array([observation]).astype(np.float32)).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        # print("ACTION: ", actions)

        # if env.is_target_reached():
        #     actions = torch.Tensor([0.0, 0.0, 0.0])

        return actions.cpu().detach().numpy().astype(np.float32)[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        print('.... successfully saved! ....')

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        print('.... successfully loaded! ....')

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # buffer에 저장된 memory 중에서 random으로 가져오기
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # 가져온 값들을 torch.tensor로 변환
        reward = torch.tensor(reward, dtype=torch.float32).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float32).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.actor.device)
        
        ''' Value function 계산 '''
        # .view(-1): return된 tensor를 1차원 vector로 펼치는 작업
        value = self.value(state).view(-1) # 현재 state에 대한 value fcn을 계산 # 현재 state에 대한 value를 return
        value_ = self.target_value(state_).view(-1) # 다음 state에 대한 목표 value fcn을 계산 # 다음 state에 대한 목표 value를 return
        value_[done] = 0.0 # episode가 종료된 상태에 대한 목표 value를 0으로 설정 # 종료되었을 때 value fcn 값은 보상이 없으므로 0

        '''
            두 개의 critic network를 사용하는 이유:
                - 안정성과 학습 성능을 향상시키기 위해서
                - 각 critic network는 다른 가중치를 가지고 있음 -> 다른 추정값을 제공
                - ensemble(앙상블) 학습과 비슷함
            둘 중 작은 값(min 값)을 선택하는 이유:
                - Robustness
                    - 각 network가 주어진 state 및 action에 대한 다른 estimation을 제공함
                    - 두 값 중에서 작은 값을 선택함으로써, network 간의 차이에 영향을 받지 않고 더 안정적인 추정을 얻을 수 있음
                - Overestimation 문제 완화
                    - 한 critic network가 다른 critic network보다 특정 상황에서 더 높은 Q값(행동 가치)을 추정할 수 있음
                    - 두 값 중에서 작은 값을 선택함으로써, overestimation 문제를 환화할 수 있음
                    - 과도한 높은 가치 추정을 방지하여 학습의 안정성과 성능을 향상시킨다
        '''
        ''' Critic network 계산 '''
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        ''' Value network 학습 '''
        self.value.optimizer.zero_grad() # 새로운 gradient descent를 위해 gradient를 초기화하는 부분
        value_target = critic_value - log_probs # Eq.3 # value target을 계산 # critic network 값에서 log 확률을 뺀 값으로, 주어진 state에서 예상되는 미래의 reward를 나타냄
        value_loss = 0.5 * F.mse_loss(value, value_target) # value network의 loss 계산 # value network의 출력과 target 간의 MSE를 계산
        value_loss.backward(retain_graph=True) # back propagation을 수행 # retain_graph: 그래프를 유지하여 여러 번의 back propagation 단계를 수행할 수 있도록 함
        self.value.optimizer.step() # parameter를 update

        ''' Critic network 계산 ''' # 왜 또 계산?
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value # Eq.12
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_ # Eq.8
        q1_old_policy = self.critic_1.forward(state, action).view(-1) # 이거는 왜 또 계산?
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat) # Eq.7
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


env = gym.make('Reacher-v4', render_mode="human")

total_num_episodes = int(5e3)
action_dim = env.action_space.shape[0] # 2
state_dim = env.observation_space.shape[0] # 11
cls_token = (151296,)
env_obs_space_tensor = torch.tensor(env.observation_space.shape)
cls_token_tensor = torch.tensor(cls_token)
input_dims_shape = torch.cat((env_obs_space_tensor, cls_token_tensor))
input_dims_shape = input_dims_shape.numpy()

# print("-------------------------", (env.observation_space.shape[0]+cls_token,))
# print("-------------------------", cls_token_tensor.shape)
# print("-------------------------", np.concatenate((env.observation_space.shape, cls_token)))

agent = SACAgent(input_dims=(151307,), env=env, n_actions=action_dim)
feature = ViTFeatureExtraction()

# vit = ViTFeatureExtraction()
# from PIL import Image
# img_path = './img/reacher.png'
# img = Image.open(img_path).convert('RGB')
# rgb_array = np.array(img)

# print("=====================\n", vit.get_feature(rgb_array).shape)


episode_durations = []
best_score = env.reward_range[0]
score_history = []
load_checkpoint = False

episode_rewards = []

env_state, _ = env.reset()
img = env.unwrapped.get_image() # rgb_array
img = feature.get_feature(img)

env_state = torch.tensor(env_state)
img_state = torch.tensor(img)

env_state = env_state.unsqueeze(0)
img_state = img_state.view(1, -1)

state = torch.cat((env_state, img_state), dim=1).view(-1)
state = state.numpy() # (151307,)

# train loop
for episode in range(MAX_EPISODES):
    env_state, _ = env.reset()
    img = env.unwrapped.get_image() # rgb_array
    img = feature.get_feature(img)

    env_state = torch.tensor(env_state)
    img_state = torch.tensor(img)

    env_state = env_state.unsqueeze(0)
    img_state = img_state.view(1, -1)

    state = torch.cat((env_state, img_state), dim=1).view(-1)
    state = state.numpy() # (151307,)
    
    episode_reward = 0

    for step in range(MAX_STEPS):
        # env_state, _ = env.reset()
        done = False

        score = 0
        
        action = agent.choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward
        
        agent.remember(state, action, reward, next_state, done)

        env_state = next_state
        img = env.unwrapped.get_image() # rgb_array
        img = feature.get_feature(img)

        env_state = torch.tensor(env_state)
        img_state = torch.tensor(img)

        env_state = env_state.unsqueeze(0)
        img_state = img_state.view(1, -1)

        state = torch.cat((env_state, img_state), dim=1).view(-1)
        state = state.numpy() # (151307,)
        
        episode_reward += reward
        
        if not load_checkpoint:
            agent.learn()
        
        if done:
            episode_durations.append(step+1)
            break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

    print('Episode: ', episode, '| Episode Reward: ', episode_reward)

    episode_rewards.append(episode_reward)

    img = env.unwrapped.get_image() # rgb_array

    if episode % 300 == 0:
        agent.save_models()

    if episode == MAX_EPISODES -1:
        agent.save_models()
        plt.show()
        break

plt.plot(episode_rewards)
plt.xlabel('Episode #')
plt.ylabel('Reward')
plt.title('Reward of Each Episode')
plt.grid(True)
# plt.ylim(-20000, 5000)
plt.show()

# test loop
# agent.load_models()

# for eps in range(10):
#     state, _ = env.reset()
#     episode_reward = 0

#     for step in range(50):
#         action = agent.choose_action(state)
#         next_state, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         env.render()

#         episode_reward += reward
#         state = next_state

#     print('Episode: ', eps, '| Episode Reward: ', episode_reward)
