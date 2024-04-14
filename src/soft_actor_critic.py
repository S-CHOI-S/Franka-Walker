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

import mujoco
import mujoco.viewer

import math
import numpy as np

import mjc_pybind.mjc_controller as mjctrl


M_PI = math.pi
M_PI_2 = M_PI/2
M_PI_4 = M_PI/4

DOF = 9

model = mujoco.MjModel.from_xml_path(os.path.abspath('../model/franka_emika_panda/pandaquest_sac.xml'))
data = mujoco.MjData(model)

controller = mjctrl.MJCController()

with mujoco.viewer.launch_passive(model, data) as viewer:

    # data.ctrl = [30,30,30,30,30,30,0,0]

    while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.

        viewer.cam.lookat = data.body('link0').subtree_com

        controller.read(data.time, data.qpos, data.qvel)
        controller.control_mujoco()
        torque = controller.write()

        for i in range(DOF - 1):
            data.ctrl[i] = torque[i]
        
        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
