# import mujoco
# import gymnasium as gym
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.distributions.normal import Normal

# from tqdm import tqdm

# JOINT = 3
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# # Value
# class ValueNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super(ValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
        
        
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


    

    
# ''' Constraints '''
# # joint angle constraint
# def get_joint_angle_constraints(model):
#     # print(mujoco.mj_name2id(model, JOINT, "foot_joint"))
#     # print(model.jnt_range[5][0])
#     return model.jnt_range[3:,:]

# def compute_constraints(log_probs, actions, constraints):
#     # Example constraint: log_probs of actions should be less than a threshold
#     constraint_violation = torch.sum(torch.clamp(log_probs - constraints, min=0.0))
#     return constraint_violation

# def compute_cost(rewards, gamma=0.99):
#     R = 0
#     cost = []
#     for r in rewards[::-1]:
#         R = r + gamma * R
#         cost.insert(0, R)
#     return cost










# # Policy (Actor)
# class PolicyNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim, max_action):
#         super(PolicyNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
        
        
        
#         self.mu = nn.Linear(128, output_dim)
#         self.sigma = nn.Linear(128, output_dim)
        
#         self.max_action = torch.tensor(max_action, dtype=torch.float32, device=DEVICE).clone().detach()
#         self.device = DEVICE
        
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
        
#         mu = torch.tanh(self.mu(x)) * torch.tensor(self.max_action).to(self.device)
#         sigma = F.softplus(self.sigma(x)) # 출력된 표준편차값에 대해서 softplus 함수를 적용해서 양수를 보장, 표준편차값 조절

#         return mu, sigma

#     def sample_action(self, state):
#         mu, sigma = self.forward(state)
#         distribution = Normal(mu, sigma)

#         actions = distribution.sample()
        
#         action = torch.tanh(actions) * torch.tensor(self.max_action, dtype=torch.float32)
        
#         log_probs = distribution.log_prob(actions) # sampling된 action에 대한 log 확률을 계산
#         log_probs = log_probs.sum(1, keepdim=True) # 각 action의 log 확률을 합산하여 반환

#         return action, log_probs
    
    

    

# class Walker():
#     def __init__(self, env=None):
#         self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0]).to(DEVICE)
#         self.value_net  = ValueNetwork(env.observation_space.shape[0]).to(DEVICE)
#         # self.critic_net = 
        
#         self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=1e-3)
#         self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        
#     def choose_action(self, state):
#         state = torch.Tensor(np.array([state]).astype(np.float32)).to(DEVICE)
#         action, action_probs = self.policy_net.sample_action(state)

#         return action.cpu().detach().numpy().astype(np.float32)[0]
    
#     # advantage (Q - V)/ GAE (Generalized Advantage Estimation) 기반 방식 사용
#     def compute_advantages(self, rewards, values, dones, gamma=0.99, tau=0.95):
#         advantage = 0
#         advantages = []
#         for i in reversed(range(len(rewards))):
#             td_error = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i] # done = 1이면 episode가 완료
#             advantage = td_error + gamma * tau * (1 - dones[i]) * advantage
#             advantages.insert(0, advantage)
#         return advantages
    
#     def update(self, states, actions, rewards, next_states, dones): # constraints도 추가해줘야 함
#         states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
#         actions = torch.tensor(actions, dtype=torch.float32).to(DEVICE)
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
#         next_states = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
#         dones = torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        
#         # advantage 계산
#         values = self.value_net(states).squeeze()
#         next_values = self.value_net(next_states).squeeze()
#         advantages = self.compute_advantages(rewards, values, dones)
#         advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        
#         _, log_probs = self.policy_net.sample_action(states)
#         policy_loss = -(log_probs * advantages).mean()
        
#         self.optimizer_policy.zero_grad()
#         policy_loss.backward()
#         self.optimizer_policy.step()
        
#         returns = values + advantages
#         value_loss = nn.MSELoss()(values, returns)
        
#         self.optimizer_value.zero_grad()
#         value_loss.backward()
#         self.optimizer_value.step()

#         # # Compute log probabilities
#         # log_probs = torch.log(self.policy_net(states))
        
#         # # Compute cost and advantages
#         # cost = compute_cost(rewards)
#         # advantages = self.compute_advantages(rewards, self.value_net(states), dones)

#         # # Compute policy loss with barrier function for constraints
#         # policy_loss = -torch.mean(log_probs * advantages)
#         # constraint_violation = compute_constraints(log_probs, actions, constraints)
#         # barrier = 1.0 / constraint_violation
        
#         # # Combine losses
#         # total_loss = policy_loss + barrier
        
#         # # Update policy network
#         # self.optimizer_policy.zero_grad()
#         # total_loss.backward()
#         # self.optimizer_policy.step()

#         # # Update value network
#         # value_loss = nn.MSELoss()(self.value_net(states), torch.tensor(cost))
#         # self.optimizer_value.zero_grad()
#         # value_loss.backward()
#         # self.optimizer_value.step()



# MAX_EPISODES = 100000
# MAX_STEPS = 1000


# def main():
#     # Walker2d-v4 환경 불러오기
#     env = gym.make('Walker2d-v4', render_mode='human')
#     model = mujoco.MjModel.from_xml_path('../model/walker2d.xml')
#     # data = mujoco.MjData(model)
    
#     # IPO Network
#     walker = Walker(env)
    
#     # num_episodes = 1000
#     gamma = 0.99
#     tau = 0.95
    
#     state, _ = env.reset()
    
#     for episode in tqdm(range(MAX_EPISODES)):
#         state, _ = env.reset()
#         episode_reward = 0
        
#         states, actions, rewards, next_states, dones = [], [], [], [], []
        
#         for step in range(MAX_STEPS):
#             done = False

#             action = walker.choose_action(state)
            
#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
            
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             next_states.append(next_state)
#             dones.append(done)
            
#             state = next_state
#             episode_reward += reward
            
#             if done:
#                 break
        
#         if episode % 300 == 0:
#             # walker.save_models()
#             pass
        
#         walker.update(states, actions, rewards, next_states, dones)
        
#         # joint_name_to_id = {name: i for i, name in enumerate(model.joint_names)}
        
#         # def get_joint_constraints(joint_name):
#         #     joint_id = joint_name_to_id[joint_name]
#         #     joint_range = model.jnt_range[joint_id]
#         #     return joint_range
        
#         # for joint_name in joint_name_to_id.keys():
#         #     joint_range = get_joint_constraints(joint_name)
#         #     print(f'Joint: {joint_name}, Range: {joint_range}')

#         # thigh_joint_range = get_joint_constraints('thigh_joint')
#         # print(f'Thigh Joint Range: {thigh_joint_range}')


# if __name__ == "__main__":
#     main()
    
    
    
    
    
    
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 10
minibatch_size = 32

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(17,128)
        self.fc_mu = nn.Linear(128,6)
        self.fc_std  = nn.Linear(128,6)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                          torch.tensor(r_batch, dtype=torch.float), torch.tensor(s_prime_batch, dtype=torch.float), \
                          torch.tensor(done_batch, dtype=torch.float), torch.tensor(prob_a_batch, dtype=torch.float)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        
def main():
    env = gym.make('Walker2d-v4', render_mode='human')
    model = PPO()
    score = 0.0
    print_interval = 20
    rollout = []

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False
        count = 0
        while count < 200 and not done:
            for t in range(rollout_len):
                mu, std = model.pi(torch.from_numpy(s).float())
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, truncated, info = env.step([a.item()])

                rollout.append((s, a, r/10.0, s_prime, log_prob.item(), done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                count += 1

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()