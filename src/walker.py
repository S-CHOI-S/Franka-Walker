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
#             print(action) # [-0.8873158   0.81476843 -0.57397985  0.19199446  0.7000201  -0.09051505]
#             print("=====================")
            
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
from collections import deque

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#learning rate backward propagation NN action
lr_actor = 0.0003
#learning rate backward propagation NN state value estimation
lr_critic = 0.0003
#Number of Learning Iteration we want to perform
Iter = 100000
#Number max of step to realise in one episode. 
MAX_STEP = 1000
#How rewards are discounted.
gamma =0.98
#How do we stabilize variance in the return computation.
lambd = 0.95
#batch to train on
batch_size = 64
# Do we want high change to be taken into account.
epsilon = 0.2
#weight decay coefficient in ADAM for state value optim.
l2_rate = 0.001


# Actor class: Used to choose actions of a continuous action space.

class Actor(nn.Module):
    def __init__(self,N_S,N_A):
      # Initialize NN structure.
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.sigma = nn.Linear(64,N_A)
        self.mu = nn.Linear(64,N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # This approach use gaussian distribution to decide actions. Could be
        # something else.
        self.distribution = torch.distributions.Normal

    def set_init(self,layers):
      # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
      # Use of tanh activation function is recommanded : bounded [-1,1],
      # gives some non-linearity, and tends to give some stability.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        # mu action output of the NN.
        mu = self.mu(x)
        #log_sigma action output of the NN
        log_sigma = self.sigma(x)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,s):
      # Choose action in the continuous action space using normal distribution
      # defined by mu and sigma of each actions returned by the NN.
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        return Pi.sample().numpy()


# Critic class : Used to estimate V(state) the state value function through a NN.
class Critic(nn.Module):
    def __init__(self,N_S):
      # Initialize NN structure.
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def set_init(self,layers):
      # Initialize weight and bias according to a normal distrib mean 0 and sd 0.1.
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
      # Use of tanh activation function is recommanded.
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        values = self.fc3(x)
        return values


class PPO:
    def __init__(self,N_S,N_A):
      # Initialize all the object we need for PPO.
        self.actor_net =Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()

    def train(self,memory):

        #Easier to hande as np array and to separate everything for each episodes.
        # 각 요소가 동일한 형태를 가지도록 np.array로 변환하기 전에 리스트로 처리
        states, actions, rewards, masks = [], [], [], []
        
        for m in memory:
            states.append(m[0])
            actions.append(m[1])
            rewards.append(m[2])
            masks.append(m[3])
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.float32)

        # Use critic network defined in Model.py
        # This function enables to get the current state value V(S).
        values = self.critic_net(states)
        # Get advantage.
        returns,advants = self.get_gae(rewards,masks,values)
        #Get old mu and std.
        old_mu,old_std = self.actor_net(states)
        #Get the old distribution.
        pi = self.actor_net.distribution(old_mu,old_std)
        #Compute old policy.
        old_log_prob = pi.log_prob(actions).sum(1,keepdim=True)

        # Everything happens here
        n = len(states)
        arr = np.arange(n)
        for epoch in range(1):
            np.random.shuffle(arr)
            for i in range(n//batch_size):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                #New parameter of the policy distribution by action.
                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(1,keepdim=True)
                old_prob = old_log_prob[b_index].detach()
                #Regularisation fixed KL : does not work as good as following clipping strategy
                # empirically.
                # KL_penalty = self.kl_divergence(old_mu[b_index],old_std[b_index],mu,std)
                ratio = torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants
                values = self.critic_net(b_states)
                # MSE Loss : (State action value - State value)^2
                critic_loss = self.critic_loss_func(values,b_returns)
                # critic_loss = critic_loss - beta*KL_penalty

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                #Clipping strategy
                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)
                clipped_loss =ratio*b_advants
                # Actual loss
                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                #Now that we have the loss, we can do the backward propagation to learn : everything is here.
                self.actor_optim.zero_grad()
                actor_loss.backward()

                self.actor_optim.step()
                
    # Get the Kullback - Leibler divergence: Measure of the diff btwn new and old policy:
    # Could be used for the objective function depending on the strategy that needs to be
    # teste.
    def kl_divergence(self,old_mu,old_sigma,mu,sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / \
             (2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)
    
    # Advantage estimation:
    def get_gae(self,rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        #Create an equivalent fullfilled of 0.
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        #Init
        running_returns = 0
        previous_value = 0
        running_advants = 0
        #Here we compute A_t the advantage.
        for t in reversed(range(0, len(rewards))):
            # Here we compute the discounted returns. Gamma is the discount factor.
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            #computes the difference between the estimated value at time step t (values.data[t]) and the discounted next value.
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            # Compute advantage
            running_advants = running_tderror + gamma * lambd * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        #Normalization to stabilize final advantage of the history to now.
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants

# Creation of a class to normalize the states
class Normalize:
    def __init__(self, N_S):
        self.mean = np.zeros((N_S,))
        self.std = np.zeros((N_S, ))
        self.stdd = np.zeros((N_S, ))
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.stdd = self.stdd + (x - old_mean) * (x - self.mean)
        if self.n > 1:
            self.std = np.sqrt(self.stdd / (self.n - 1))
        else:
            self.std = self.mean

        x = x - self.mean
        x = x / (self.std + 1e-8)
        x = np.clip(x, -5, +5)
        return x


env = gym.make('Walker2d-v4', render_mode='human')

#Number of state and action
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

# Random seed initialization
# env.seed(500)
# torch.manual_seed(500)
# np.random.seed(500)

# Run the Ppo class
frames = []
ppo = PPO(N_S,N_A)
# Normalisation for stability, fast convergence... always good to do.
normalize = Normalize(N_S)
episodes = 0
eva_episodes = 0
state, _ = env.reset()

for iter in tqdm(range(Iter)):
    memory = deque()
    scores = []
    steps = 0
    while steps < 2048: #Horizon
        episodes += 1
        state, _ = env.reset()
        s = normalize(state)
        score = 0
        for _ in range(MAX_STEP):
            steps += 1
            #Choose an action: detailed in PPO.py
            # The action is a numpy array of 17 elements. It means that in the 17 possible directions of action we have a specific value in the continuous space.
            # Exemple : the first coordinate correspond to the Torque applied on the hinge in the y-coordinate of the abdomen: this is continuous space.
            a=ppo.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]

            #Environnement reaction to the action : There is a reaction in the 376 elements that characterize the space :
            # Exemple : the first coordinate of the states is the z-coordinate of the torso (centre) and using env.step(a), we get the reaction of this state and
            # of all the other ones after the action has been made.
            s_ , r ,truncated, terminated ,info = env.step(a)
            done = truncated or terminated
            s_ = normalize(s_)

            # Do we continue or do we terminate an episode?
            mask = (1-done)*1
            memory.append([s,a,r,mask])
            # print('s: ', s)
            # print('a: ', a)
            # print('r: ', r)
            # print('mask: ', mask)
            
            score += r
            s = s_
            if done:
                break
        # with open('log_' + args.env_name  + '.txt', 'a') as outfile:
        #     outfile.write('\t' + str(episodes)  + '\t' + str(score) + '\n')
        scores.append(score)
    score_avg = np.mean(scores)
    print('{} episode score is {:.2f}'.format(episodes, score_avg))
    ppo.train(memory)















# # Hyperparameters
# learning_rate = 0.0003
# gamma = 0.9
# lmbda = 0.9
# eps_clip = 0.2
# K_epoch = 10
# rollout_len = 3
# buffer_size = 10
# minibatch_size = 32

# class PPO(nn.Module):
#     def __init__(self):
#         super(PPO, self).__init__()
#         self.data = []

#         self.fc1 = nn.Linear(17, 128)
#         self.fc_mu = nn.Linear(128, 6)
#         self.fc_std = nn.Linear(128, 6)
#         self.fc_v = nn.Linear(128, 1)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         self.optimization_step = 0

#     def pi(self, x, softmax_dim=0):
#         x = F.relu(self.fc1(x))
#         mu = 2.0 * torch.tanh(self.fc_mu(x))
#         std = F.softplus(self.fc_std(x))
#         return mu, std

#     def v(self, x):
#         x = F.relu(self.fc1(x))
#         v = self.fc_v(x)
#         return v

#     def put_data(self, transition):
#         print(f"{YELLOW}transition: \n{RESET}", transition)
#         self.data.append(transition)

#     def make_batch(self):
#         s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
#         data = []

#         for j in range(buffer_size):
#             s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
#             for i in range(minibatch_size):
#                 rollout = self.data.pop()

#                 for transition in rollout:
#                     s, a, r, s_prime, prob_a, done = transition

#                     s_lst.append(s)
#                     a_lst.append(a)
#                     r_lst.append(r)
#                     s_prime_lst.append(s_prime)
#                     prob_a_lst.append(prob_a)
#                     done_mask = 0.0 if done else 1.0
#                     done_lst.append(done_mask)

#                 s_batch.append(s_lst)
#                 a_batch.append(a_lst)
#                 r_batch.append(r_lst)
#                 s_prime_batch.append(s_prime_lst)
#                 prob_a_batch.append(prob_a_lst)
#                 done_batch.append(done_lst)

#             s_batch_tensor = torch.tensor(s_batch, dtype=torch.float).view(-1, 17)
#             a_batch_tensor = torch.tensor(a_batch, dtype=torch.float).view(-1, 6)
#             r_batch_tensor = torch.tensor(r_batch, dtype=torch.float).view(-1, 1)
#             s_prime_batch_tensor = torch.tensor(s_prime_batch, dtype=torch.float).view(-1, 17)
#             prob_a_batch_tensor = torch.tensor(prob_a_batch, dtype=torch.float).view(-1, 6)
#             done_batch_tensor = torch.tensor(done_batch, dtype=torch.float).view(-1, 1)

#             mini_batch = (s_batch_tensor, a_batch_tensor, r_batch_tensor,
#                           s_prime_batch_tensor, done_batch_tensor, prob_a_batch_tensor)

#             data.append(mini_batch)

#         return data

#     def calc_advantage(self, data):
#         data_with_adv = []
#         for mini_batch in data:
#             s, a, r, s_prime, done_mask, old_log_prob = mini_batch
#             with torch.no_grad():
#                 td_target = r + gamma * self.v(s_prime).squeeze(-1) * done_mask
#                 delta = td_target - self.v(s).squeeze(-1)
#             delta = delta.numpy()

#             advantage_lst = []
#             advantage = 0.0
#             for delta_t in delta[::-1]:
#                 advantage = gamma * lmbda * advantage + delta_t[0]
#                 advantage_lst.append([advantage])
#             advantage_lst.reverse()
#             advantage = torch.tensor(advantage_lst, dtype=torch.float)
#             advantage = advantage.view(-1, 1)  # 여기에 차원 맞추기 추가
#             data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

#         return data_with_adv

#     def train_net(self):
#         if len(self.data) == minibatch_size * buffer_size:
#             data = self.make_batch()
#             data = self.calc_advantage(data)

#             for i in range(K_epoch):
#                 for mini_batch in data:
#                     s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

#                     mu, std = self.pi(s, softmax_dim=1)
#                     dist = Normal(mu, std)
#                     log_prob = dist.log_prob(a)
#                     ratio = torch.exp(log_prob.sum(dim=1, keepdim=True) - old_log_prob.sum(dim=1, keepdim=True))  # 차원 맞추기 추가

#                     surr1 = ratio * advantage
#                     surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
#                     loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s).squeeze(-1), td_target)

#                     self.optimizer.zero_grad()
#                     loss.mean().backward()
#                     nn.utils.clip_grad_norm_(self.parameters(), 1.0)
#                     self.optimizer.step()
#                     self.optimization_step += 1

# def main():
#     env = gym.make('Walker2d-v4', render_mode='human')
#     model = PPO()
#     score = 0.0
#     print_interval = 20
#     rollout = []

#     for n_epi in tqdm(range(10000)):
#         s, _ = env.reset()
#         done = False
#         count = 0
#         while count < 200 and not done:
#             for t in range(rollout_len):
#                 mu, std = model.pi(torch.from_numpy(s).float())
#                 dist = Normal(mu, std)
#                 a = dist.sample()
#                 log_prob = dist.log_prob(a)

#                 a = a.cpu().detach().numpy()
#                 print(f"{RED}log_prob: {RESET}\n", log_prob)

#                 s_prime, r, done, truncated, info = env.step(a)

#                 rollout.append((s, a, r / 10.0, s_prime, log_prob, done))
#                 if len(rollout) == rollout_len:
#                     model.put_data(rollout)
#                     rollout = []

#                 s = s_prime
#                 score += r
#                 count += 1

#             model.train_net()

#         if n_epi % print_interval == 0 and n_epi != 0:
#             print("# of episode :{}, avg score : {:.1f}, optmization step: {}".format(n_epi, score / print_interval, model.optimization_step))
#             score = 0.0

#     env.close()

# if __name__ == '__main__':
#     main()
