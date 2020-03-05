# Modified from Pytorch DQN Tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# by Adam Young youngcadam@ucla.edu 

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used


torch.set_default_tensor_type(torch.cuda.FloatTensor) 

env = gym.make('CartPole-v1').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
####################### Experience Replay ############################
######################################################################
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
############################### DQN ##################################
######################################################################
class DQN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) #how to apply linear activation
        return x


BATCH_SIZE = 32
GAMMA = 0.95
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = .995
TARGET_UPDATE = 1
epsilon = EPS_START

n_actions = env.action_space.n

policy_net = DQN().to(device)
# target_net = DQN().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr = .001)
memory = ReplayMemory(2000)


steps_done = 0


def select_action(state):
    if np.random.rand() <= epsilon:
        return torch.tensor([random.randrange(n_actions)])
    else:
        state=state.to(device)
        with torch.no_grad():
            if policy_net(state)[0] > policy_net(state)[1]:
                return torch.tensor([0])
            else:
                return torch.tensor([1])
        

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training: input = (1,4) state vector ')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.cpu().numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.cpu().numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
 
    minibatch = memory.sample(BATCH_SIZE) #list of the form
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + GAMMA * np.amax(policy_net(next_state).cpu().detach().numpy()))
        Q_target = policy_net(state)[action]
        loss = F.mse_loss(Q_target, target)        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
    global epsilon
    if epsilon > EPS_END:
        epsilon *= EPS_DECAY



# Main training loop
num_episodes = 2000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float)
    state = state.to(device)  
    for t in count():
        # Select and perform an action
        action = select_action(state)
        newstate, reward, done, _ = env.step(action.item())
        newstate = torch.tensor(newstate, dtype=torch.float)
        newstate = newstate.to(device)

        reward = torch.tensor([reward], device=device)

        if not done:
            next_state =  newstate 
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, reward, next_state, done) 

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)

        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
#    if i_episode % TARGET_UPDATE == 0:
#        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()

plt.savefig('/figures/state_input_results.png')
