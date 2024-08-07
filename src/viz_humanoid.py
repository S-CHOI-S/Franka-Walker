import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

from humanoid import PPO, Normalize

from color_code import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make('Humanoid-v4', render_mode='human')

log_dir = "../runs/humanoid/20240807_13-49-06/"

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

average_constraint_limit = []
normalize_avg_cstrnt_limit = lambda x, min_val, max_val: 2 * (x - min_val) / (max_val - min_val) - 1
average_constraint_limit.append(normalize_avg_cstrnt_limit(0.2, -0.785, 0.785)) # (-0.785, 0.785)
average_constraint_limit.append(normalize_avg_cstrnt_limit(-0.2, -0.785, 0.785))
average_constraint_limit.append(normalize_avg_cstrnt_limit(0.15, -1.31, 0.524)) # (-1.31, 0.524)
average_constraint_limit.append(normalize_avg_cstrnt_limit(-0.15, -1.31, 0.524))
average_constraint_limit.append(normalize_avg_cstrnt_limit(0.13, -0.611, 0.611)) # (-0.611, 0.611)
average_constraint_limit.append(normalize_avg_cstrnt_limit(-0.13, -0.611, 0.611))
print(f"{BLUE}avg_cstrnt_limit_normalized: {RESET}\n{average_constraint_limit}")

# Initialize PPO model
# ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=0)
ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=6, avg_cstrnt_limit=average_constraint_limit)
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
    avg_cstrnt1 = []
    avg_cstrnt2 = []
    avg_cstrnt3 = []
    for _ in range(1000):
        action = ppo.actor_net.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        state = normalize(state)
        score += reward
        # print(f"{BLUE}z angle: {RESET}{state[5]}, {BLUE}y angle: {RESET}{state[6]}, {BLUE}x angle: {RESET}{state[7]}")

        avg_cstrnt1.append(state[5])
        avg_cstrnt2.append(state[6])
        avg_cstrnt3.append(state[7])

        if done:
            break

    cstrnt1_avg = np.mean(avg_cstrnt1)
    cstrnt2_avg = np.mean(avg_cstrnt2)
    cstrnt3_avg = np.mean(avg_cstrnt3)

    print(f"\nEPISODE NUM: {episode_id} ========================================================")
    print(f"{BLUE}z-limit: {RESET}{average_constraint_limit[1]:.3f}/{average_constraint_limit[0]:.3f},\t{BLUE}y-limit: {RESET}{average_constraint_limit[3]:.3f}/{average_constraint_limit[2]:.3f},\t"
          f"{BLUE}x-limit: {RESET}{average_constraint_limit[5]:.3f}/{average_constraint_limit[4]:.3f}")
    print(f"{CYAN}z-angle: {RESET}{cstrnt1_avg:.3f},\t{CYAN}y-angle: {RESET}{cstrnt2_avg:.3f},\t{CYAN}x-angle: {RESET}{cstrnt3_avg:.3f}")
        
env.close()