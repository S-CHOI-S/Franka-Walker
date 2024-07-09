import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import deque
from tqdm import tqdm

from walker import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(env, model, episodes=10):
    scores = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = model.actor_net.choose_action(state)
            action = action.squeeze()
            state, reward, done, _, _ = env.step(action)
            env.render()
            
            total_reward += reward
        scores.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    print(f"Average Reward over {episodes} episodes: {np.mean(scores)}")
    env.close()

# Initialize environment
env = gym.make('Walker2d-v4', render_mode='human')

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Initialize PPO model
ppo = PPO(N_S, N_A)

# Load the saved model
log_dir = "../runs/20240708_11-19-08/ppo/51300"
ppo.load(log_dir)
# ppo.eval()

state, _ = env.reset()
test_model(env, ppo, episodes=10)
