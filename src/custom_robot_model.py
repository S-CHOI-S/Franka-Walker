import os
import time

import mujoco
import mujoco.viewer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98

# Policy
class Policy(nn.Module):
	def __init__(self): # policy network에 필요한 연산들을 정의
			super(Policy, self).__init__()
			self.data = []
			
			self.fc1 = nn.Linear(4, 128) # 길이 4인 vector가 input으로 들어가
			self.fc2 = nn.Linear(128, 2) # 2개의 action에 대한 확률값을 return
			self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
			
	def forward(self, x): # neural net 구조를 만들어줌
		# Linear(4,128)
		# relu ( activation 함수, 음수 값은 0으로, 양수 값은 비례해서 나타냄 )
		# Linear(128,2)
		# softmax ( activation 함수, -inf, +inf로 가면 각각 하나의 값으로 수렴하도록 )
			# 각 class에 해당될 확률을 뽑아낼 수 있음
			# dim = 0: row들을 기준으로 softmax를 계산한다는 의미
		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim = 0)
		return x
		
	def put_data(self, item):
		self.data.append(item)
		
	def train_net(self): # 실제로 network를 학습하는 코드
		R = 0
		self.optimizer.zero_grad()
		for r, prob in self.data[::-1]:
			# data[::-1] -> data의 모든 요소를 역순으로 가져온다
			R = r + gamma * R
			loss = -R * torch.log(prob)
			loss.backward() # loss에 대한 gradient가 계산되어 더해짐
		self.optimizer.step() # 축적된 gradient를 이용해 neural net의 parameter update
		self.data = []

# Agent



pi = Policy()
####################################################################################
####################################################################################

# m = mujoco.MjModel.from_xml_path(os.path.abspath('../model/panda.xml')) # should run this node in src directory
m = mujoco.MjModel.from_xml_path(os.path.abspath('../model/franka_emika_panda/scene_valve.xml')) # should run this node in src directory
d = mujoco.MjData(m)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(m)

# print(m.ngeom) # geom의 개수: 126
# print(m.geom_rgba)

# try:
#   m.geom()
# except KeyError as e:
#   print(e)

print("A")

with mujoco.viewer.launch_passive(m, d) as viewer:
   # close the viewer automatically after 30 wall-seconds!
   start = time.time()
   while viewer.is_running(): # and time.time() - start < 30:
      step_start = time.time()
      mujoco.mj_step(m, d)
      # 시간이 종료되면 'Segmentation fault (core dumped)'라고 terminal에 출력된다!
   print("B")
   # mj_step can be replaced with code that also evaluates
   # a policy and applies a control signal before stepping the physics
   
   # mujoco.mj_step(m, d)
   # print(d.xpos)
   # print("C")
   # print(d.geom_xpos)
   
   # mujoco.mj_kinematics(m, d)
   # print('raw access:\n', d.geom_xpos)

   # # MjData also supports named access:
   # print('\nnamed access:\n', d.geom('handle_base').xpos)

   # example modification of a viewer option: toggle contact points every two seconds
   with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

   # pick up changes to the physics state, apply perturbations, update options from GUI
   viewer.sync()

   # rudimentary time keeping, will drift relative to wall clock
   time_until_next_step = m.opt.timestep - (time.time() - step_start)
   if time_until_next_step > 0:
      time.sleep(time_until_next_step)
