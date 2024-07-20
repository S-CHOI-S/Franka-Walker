import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from collections import deque
from tqdm import tqdm

from walker import PPO, Normalize

from color_code import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = gym.make('Walker2d-v4', render_mode='human')

# log_dir = "../runs/20240715_19-42-33/"
log_dir = "../runs/20240716_14-00-55/"

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Initialize PPO model
ppo = PPO(N_S, N_A, log_dir)
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
    cstrnt1 = []
    cstrnt2 = []
    for _ in range(1000):
        action = ppo.actor_net.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        state = normalize(state)
        score += reward
        
        cstrnt1.append(state[1])
        cstrnt2.append(state[8])
        
        # if (state[1] <= 0.2) & (state[8] <= 0.5):
        #     print(f"\ny_angle_of_the_torso is {GREEN}{state[1]:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{state[8]:.3f}{RESET}")
        # elif (state[1] <= 0.2) & (state[8] > 0.5):
        #     print(f"\ny_angle_of_the_torso is {GREEN}{state[1]:.3f}{RESET}, x_vel_of_the_torso is {RED}{state[8]:.3f}{RESET}")
        # elif (state[1] > 0.2) & (state[8] <= 0.5):
        #     print(f"\ny_angle_of_the_torso is {RED}{state[1]:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{state[8]:.3f}{RESET}")
        # else:
        #     print(f"\ny_angle_of_the_torso is {RED}{state[1]:.3f}{RESET}, x_vel_of_the_torso is {RED}{state[8]:.3f}{RESET}")

        if done:
            break
    
    cstrnt1_avg = np.mean(cstrnt1)
    cstrnt2_avg = np.mean(cstrnt2)
    
    if (cstrnt1_avg <= 0.2) & (cstrnt2_avg <= 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {GREEN}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{cstrnt2_avg:.3f}{RESET}")
    elif (cstrnt1_avg <= 0.2) & (cstrnt2_avg > 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {GREEN}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {RED}{cstrnt2_avg:.3f}{RESET}")
    elif (cstrnt1_avg > 0.2) & (cstrnt2_avg <= 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {RED}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{cstrnt2_avg:.3f}{RESET}")
    else:
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {RED}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {RED}{cstrnt2_avg:.3f}{RESET}")
env.close()