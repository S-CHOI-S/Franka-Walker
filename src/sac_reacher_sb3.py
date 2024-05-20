import gymnasium as gym

from stable_baselines3 import SAC

env = gym.make("Reacher-v4", render_mode="human")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e6, log_interval=4)
model.train(batch_size=256)
model.save("sac_reacher_stbl")

# del model # remove to demonstrate saving and loading

model = SAC.load("sac_reacher_stbl")

obs, info = env.reset()

episode_rewards = []
episode_reward = 0

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    print(episode_reward)
    if terminated or truncated:
        print(episode_reward)
        episode_rewards.append(episode_reward)
        episode_reward = 0
        obs, info = env.reset()
    

