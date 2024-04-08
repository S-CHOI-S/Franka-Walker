# import gymnasium as gym

# from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1", render_mode="human")

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("sac_pendulum")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()

#################################################################################
import gymnasium as gym
from stable_baselines3 import SAC

import os
import time

import mujoco
import mujoco.viewer

class PandaEnv(gym.Env):
    metadata = {"render_nodes": ["human"]}

    def __init__(self, render_mode="human"):
        super(PandaEnv, self).__init__()
        self.render_mode = render_mode

        self.m = mujoco.MjModel.from_xml_path(os.path.abspath('../model/pandaquest.xml'))
        self.d = mujoco.MjData(self.m)

        renderer = mujoco.Renderer(self.m)

        print("__init__ function successed!")

    def load_mjc_viewer(self):
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            start = time.time()
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(self.m, self.d)

                self.load_current_states()
            
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.d.time % 2)

            viewer.sync()

            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        print("load_mjc_viewer function successed!")

    def load_current_states(self):
        print(self.d.xpos) # 각 링크의 위치를 나타낸다 -> ?
        print(self.d.qpos) # 각 관절의 각도를 나타낸다

        print("load_current_states function successed!")




env = PandaEnv()
env.load_mjc_viewer()
# env.load_current_states()