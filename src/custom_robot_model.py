import os
import time

import mujoco
import mujoco.viewer

# m = mujoco.MjModel.from_xml_path(os.path.abspath('../model/panda.xml')) # should run this node in src directory
m = mujoco.MjModel.from_xml_path(os.path.abspath('../model/franka_emika_panda/scene_valve.xml')) # should run this node in src directory
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
   # close the viewer automatically after 30 wall-seconds!
   start = time.time()
   while viewer.is_running() and time.time() - start < 30:
      step_start = time.time()
      # 시간이 종료되면 'Segmentation fault (core dumped)'라고 terminal에 출력된다!

   # mj_step can be replaced with code that also evaluates
   # a policy and applies a control signal before stepping the physics

   mujoco.mj_step(m, d)

   # example modification of a viewer option: toggle contact points every two seconds
   with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

   # pick up changes to the physics state, apply perturbations, update options from GUI
   viewer.sync()

   # rudimentary time keeping, will drift relative to wall clock
   time_until_next_step = m.opt.timestep - (time.time() - step_start)
   if time_until_next_step > 0:
      time.sleep(time_until_next_step)
