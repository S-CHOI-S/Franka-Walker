# # import gymnasium as gym

# # from stable_baselines3 import SAC

# # env = gym.make("Pendulum-v1", render_mode="human")

# # model = SAC("MlpPolicy", env, verbose=1)
# # model.learn(total_timesteps=10000, log_interval=4)
# # model.save("sac_pendulum")

# # del model # remove to demonstrate saving and loading

# # model = SAC.load("sac_pendulum")

# # obs, info = env.reset()
# # while True:
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, reward, terminated, truncated, info = env.step(action)
# #     if terminated or truncated:
# #         obs, info = env.reset()

# #################################################################################
# import gymnasium as gym
# from stable_baselines3 import SAC

import os
import time

import gymnasium as gym
from gym import spaces

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

# define
M_PI = math.pi
M_PI_2 = M_PI/2
M_PI_4 = M_PI/4

DOF = 9

MAX_EPISODES = 100000
MAX_STEPS = 100000

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
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

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
    def __init__(self, learning_rate, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
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
    def __init__(self, learning_rate, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
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
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
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

        mu = torch.tanh(self.mu(prob)) * torch.tensor(self.max_action).to(self.device)
        sigma = F.softplus(self.sigma(prob))

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class SACAgent():
    def __init__(self, alpha=0.0003, learning_rate=0.0003, input_dims=[3],
            env=None, gamma=0.99, n_actions=3, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high

        self.actor = ActorNetwork(alpha, input_dims, n_actions=self.max_action,
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
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

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

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

class Environment:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath('../model/franka_emika_panda/pandaquest_sac.xml'))
        self.data = mujoco.MjData(self.model)

        self.controller = mjctrl.MJCController()

        self.goal_position = [0.25, -0.2, 0.8]
        self.current_position = self.data.xpos[mujoco.mj_name2id(self.model, 1, "link7")]

        # Initialize stack attribute
        self._k = 7
        self.stack = 5

        # self.action_space = self._construct_action_space()
        # self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()
        self.observation_space = self._construct_observation_space()

    def _construct_action_space(self):
        action_space = 3
        action_low = -1 * np.ones(action_space)
        action_high = 1 * np.ones(action_space)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    def _construct_observation_space(self):
        s = {'object': spaces.Box(shape=(1, 14), low=-np.inf, high=np.inf, dtype=np.float32),
             'q': spaces.Box(shape=(self.stack, self._k), low=-1, high=1, dtype=np.float32),
             'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
             'rpy_des': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
             'x_plan': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
             'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            }
        return spaces.Dict(s)

    def reset(self):
        self.current_position = self.data.xpos[mujoco.mj_name2id(self.model, 1, "link7")]
        self.steps = 0
        return self.current_position

    def step(self, action):

        self.current_position += action

        self.controller.read(self.data.time, self.data.qpos, self.data.qvel)
        self.controller.control_mujoco()
        self.torque = self.controller.write()

        for i in range(DOF - 1):
            self.data.ctrl[i] = self.torque[i]
        
        mujoco.mj_step(self.model, self.data)

        reward = self.calculate_reward()
        done = self.is_done()

        self.steps += 1
        # self.data.ctrl[:] = action
        # self.sim.step()
        # observation = self.sim.get_state()
        # reward = self.calculate_reward()
        # done = self.is_done()
        # info = {}  # Additional information (optional)
        # return observation, reward, done, info

        return self.current_position, reward, done, {}
    
    def calculate_reward(self):
        # Calculate reward based on current state
        distance_to_goal = np.linalg.norm(self.current_position - self.goal_position)
        reward = -distance_to_goal  # Example: negative distance as reward
        return reward
    
    def is_done(self):
        # Determine if the episode is done
        if self.steps >= MAX_STEPS:  # Example: episode ends after a certain number of steps
            return True
        if np.allclose(self.current_position, self.goal_position, atol=1e-3):  # Example: episode ends when goal is reached
            return True
        return False


# Define hyperparameters
alpha = 0.0003
beta = 0.0003
input_dims = [3]  # Dimension of Cartesian coordinates [x, y, z]
gamma = 0.99
tau = 0.005
batch_size = 256
reward_scale = 2


# __main__
# model = mujoco.MjModel.from_xml_path(os.path.abspath('../model/franka_emika_panda/pandaquest_sac.xml'))
# data = mujoco.MjData(model)

# controller = mjctrl.MJCController()

# goal_position = [0.25, -0.2, 0.8]
# current_position = data.xpos[mujoco.mj_name2id(model, 1, "link7")]

env = Environment()
agent = SACAgent(env=env)

for episode in range(MAX_EPISODES):
    observation = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

            # data.ctrl = [30,30,30,30,30,30,0,0]

            while viewer.is_running():
                step_start = time.time()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.

                # for camera tracking
                viewer.cam.lookat = env.data.body('link0').subtree_com
                viewer.cam.elevation = -15

                current_position = env.data.xpos[mujoco.mj_name2id(env.model, 1, "link7")]

                action = agent.choose_action(observation)

                # controller.read(data.time, data.qpos, data.qvel)
                # controller.control_mujoco()
                # torque = controller.write()

                # for i in range(DOF - 1):
                #     data.ctrl[i] = torque[i]
                
                # mujoco.mj_step(model, data)

                new_observation, reward, done, _ = env.step(action)

                agent.remember(observation, action, reward, new_observation, done)

                agent.learn()

                observation = new_observation
                episode_reward += reward
                print("action: ", action)
                

                if done:
                    break       

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
