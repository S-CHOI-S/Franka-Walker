import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import deque
from tqdm import tqdm

from walker import PPO, Normalize

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make('Walker2d-v4', render_mode='human')

log_dir = "../runs/20240715_16-33-43/"

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Initialize PPO model
ppo = PPO(N_S, N_A, log_dir)
normalize = Normalize(N_S, log_dir, train_mode=False)

# Load the saved model
ppo.actor_net.load_model()
ppo.actor_net.eval()
normalize.load_params()

# Test the model
state, _ = env.reset()
state = normalize(state)

test_total_reward = 0
test_episodes = 10  # Number of episodes to test
for episode_id in range(test_episodes):
    state, _ = env.reset()
    state = normalize(state)
    score = 0
    for _ in range(1000):
        action = ppo.actor_net.choose_action(state)
        # print(f"{YELLOW}walker velocity: {RESET}", state[8])
        state, reward, done, _, _ = env.step(action)
        state = normalize(state)
        score += reward
        # print(f"{MAGENTA}thigh_left_joint: {RESET}", state[5])

        if done:
            env.reset()
            break
    
    print("episode: ", episode_id, "\tscore: ", score)
env.close()