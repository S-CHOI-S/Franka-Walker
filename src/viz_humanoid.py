import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from walker import PPO, Normalize

from color_code import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make('Humanoid-v4', render_mode='human')

log_dir = "../runs/humanoid/20240806_15-43-25/"

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Initialize PPO model
ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=0)
normalize = Normalize(N_S, log_dir, train_mode=False)

# Load the saved model
ppo.actor_net.load_model()
ppo.actor_net.eval()
ppo.critic_net.load_model()
ppo.critic_net.eval()
ppo.load()
normalize.load_params()

# Test the model
test_total_reward = 0
test_episodes = 10  # Number of episodes to test
for episode_id in range(test_episodes):
    state, _ = env.reset()
    state = normalize(state)
    score = 0
    for _ in range(1000):
        action = ppo.actor_net.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        state = normalize(state)
        score += reward

        if done:
            break
        
env.close()