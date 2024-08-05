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
env = gym.make('Walker2d-v4', render_mode='human')

# log_dir = "../runs/20240715_19-42-33/"
# file_dir1 = "/home/kist/franka_walker/runs/20240725_19-01-40/"
# log_dir = "../runs/20240725_15-23-19/" # 2 constraints
log_dir = "../runs/20240805_16-55-17/" # 4 constraints

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

cstrnt1_limit = 0.2 # y angle of the torso
cstrnt2_limit = 0.5 # x vel of the torso
cstrnt3_limit = 1
cstrnt4_limit = 1

# Initialize PPO model
# ppo = PPO(N_S, N_A, log_dir, num_constraints=2, cstrnt_limit=[cstrnt1_limit, cstrnt2_limit])
ppo = PPO(N_S, N_A, log_dir, num_avg_constraints=1, avg_cstrnt_limit=[cstrnt2_limit])
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
    cstrnt3 = []
    cstrnt4 = []
    for _ in range(1000):
        action = ppo.actor_net.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        state = normalize(state)
        score += reward
        
        cstrnt1.append(state[1])
        cstrnt2.append(state[8])
        cstrnt3.append(-state[3])
        cstrnt4.append(-state[6])

        if state[1] > 0.2:
            print(f"{RED}state[1]: {RESET}", state[1])
        
        # print(f"{RESET}angle of the thigh joint:      {RESET}", state[2])
        # print(f"{MAGENTA}angle of the leg joint:        {RESET}", state[3]) ## leg를 -1보다 크게
        # print(f"{RESET}angle of the left thigh joint: {RESET}", state[5])
        # print(f"{MAGENTA}angle of the left leg joint:   {RESET}", state[6]) ## leg를 -1보다 크게
        # print("==============================================================")
        
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
    cstrnt3_avg = np.mean(cstrnt3)
    cstrnt4_avg = np.mean(cstrnt4)
    
    if (cstrnt1_avg <= 0.2) & (cstrnt2_avg <= 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {GREEN}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{cstrnt2_avg:.3f}{RESET}")
    elif (cstrnt1_avg <= 0.2) & (cstrnt2_avg > 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {GREEN}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {RED}{cstrnt2_avg:.3f}{RESET}")
    elif (cstrnt1_avg > 0.2) & (cstrnt2_avg <= 0.5):
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {RED}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {GREEN}{cstrnt2_avg:.3f}{RESET}")
    else:
        print(f"\nepisode: {episode_id}, score: {score}, y_angle_of_the_torso is {RED}{cstrnt1_avg:.3f}{RESET}, x_vel_of_the_torso is {RED}{cstrnt2_avg:.3f}{RESET}")
        
    if (cstrnt3_avg <= 1) & (cstrnt4_avg <= 1):
        print(f"\n\t\t\t\tcstrnt3 is {GREEN}{cstrnt3_avg:.3f}{RESET}, cstrnt4 is {GREEN}{cstrnt4_avg:.3f}{RESET}")
    elif (cstrnt3_avg <= 1) & (cstrnt4_avg > 1):
        print(f"\n\t\t\tcstrnt3 is {GREEN}{cstrnt3_avg:.3f}{RESET}, cstrnt4 is {RED}{cstrnt4_avg:.3f}{RESET}")
    elif (cstrnt3_avg > 1) & (cstrnt4_avg <= 1):
        print(f"\n\t\t\tcstrnt3 is {RED}{cstrnt3_avg:.3f}{RESET}, cstrnt4 is {GREEN}{cstrnt4_avg:.3f}{RESET}")
    else:
        print(f"\n\t\t\tcstrnt3 is {RED}{cstrnt3_avg:.3f}{RESET}, cstrnt4 is {RED}{cstrnt4_avg:.3f}{RESET}")
        
env.close()