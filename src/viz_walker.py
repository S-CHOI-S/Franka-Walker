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

log_dir = "../runs/20240711_15-21-10/"

# Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Initialize PPO model
ppo = PPO(N_S, N_A, log_dir)
normalize = Normalize(N_S)

# Load the saved model
# ppo.load(log_dir)
ppo.actor_net.load_model()
# normalize.load_params(log_dir + "../../normalize_params.npy")
ppo.actor_net.eval()

# Test the model
now_state, _ = env.reset()
now_state = normalize(now_state)

test_total_reward = 0
test_episodes = 10  # Number of episodes to test
for episode_id in range(test_episodes):
    now_state, _ = env.reset()
    now_state = normalize(now_state)
    score = 0
    for _ in range(1000):
        #with torch.no_grad():
            #ppo.actor_net.eval()
        a = ppo.actor_net.choose_action(torch.from_numpy(np.array(now_state).astype(np.float32)).unsqueeze(0))[0]
        # print(f"{YELLOW}walker velocity: {RESET}", now_state[8])
        now_state, r, done, _, _ = env.step(a)
        now_state = normalize(now_state)
        score += r

        if done:
            env.reset()
            break
    print("episode: ", episode_id, "\tscore: ", score)
env.close()
# for _ in range(test_episodes):
#     state, _ = env.reset()
#     state = normalize(state)
#     done = False
#     episode_reward = 0
#     while not done:
#         action = ppo.actor_net.choose_action(torch.from_numpy(np.array(state).astype(np.float32)).unsqueeze(0))[0]
#         next_state, reward, truncated, terminated, info = env.step(action)
#         episode_reward += reward
#         state = normalize(next_state)
#         done = truncated or terminated
#     test_total_reward += episode_reward
# average_test_reward = test_total_reward / test_episodes
# print('Average test reward: {:.2f}'.format(average_test_reward))
