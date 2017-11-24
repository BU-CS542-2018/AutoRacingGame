'''
Created on Nov 11, 2017

@author: micou
'''
import gym
import universe  # register the universe environments
import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import csv
import time
from PIL import Image
from itertools import count

from DQN import *
from statsmodels import duration

RESUME = True
filename = "./local_training.pth.tar"

# Some global variables
BATCH_SIZE = 128 # used for experience pool based training
GAMMA = 0.999 # Used for expectation decay
EPS_START = 0.9 # using epsilon-greedy algorithm, with this probability to take random step
EPS_END = 0.05 # At the end of training, probability to take random step is quite small
EPS_DECAY = 20000 # Epsilon decay in exponential speed. Set the decay speed
output_threshold = 0.5

# Use model set at DQN.py, model have 4 output
if not RESUME:
    model = DQN()
    if use_cuda:
        model.cuda()
    episode_duration = []
    memory = ReplayMemory(10000)
    steps_done = 0
else:
    checkpoint = torch.load(filename)
    model = DQN()
    if use_cuda:
        model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    episode_duration = checkpoint['episode_duration']
    memory = checkpoint['memory']
    steps_done = checkpoint['steps_done']
    print("Loaded model, start from episode ", len(episode_duration))

last_sync = 0
optimizer = optim.RMSprop(model.parameters())
RESIZE = T.Compose([T.ToPILImage(),
                    T.Scale(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def select_action(state):
    '''
    input state and use epsilon-greedy algorithm to select actions to take
    INPUT: state matrix
    OUTPUT: [] of 0s and 1s. list length is number of actions. Each state map to one action
    '''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        result = model(Variable(state, volatile=True).type(FloatTensor)).data
        max_index_result = result.max(1)[1].view(1, 1)
        return max_index_result
        
        # Only cpu can do threshold
#         if use_cuda:
#             result = result.cpu().numpy()
#             result = np.where(result>=output_threshold, 1, 0)
#             return torch.from_numpy(result).cuda()
#         else:
#             result = result.numpy()
#             result = np.where(result>=output_threshold, 1, 0)
#             return torch.from_numpy(result)
    else:
        return LongTensor([[random.randrange(ACTION_NUM)]])

def plot_durations():
    '''
    Plot training result
    '''
    plt.figure(1)
    plt.clf()
    durations_t = torch.FloatTensor(episode_duration)
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        print(durations_t)
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)

def optimize_model():
    """ERROR"""
    global last_sync
    if len(memory) < BATCH_SIZE:
        # Waiting until there are enough sample to update model
        return 
    last_sync += 1
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    # Freeze network. 
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile = True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    
    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values = next_state_values.view(-1, 1)
    next_state_values.volatile = False
    
    # Update network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
def get_screen():
    screen = env.render(mode="rgb_array")
    # cut out the game image
    screen = screen[84:596, 18:818]
    # Transpose the image
    screen = screen.transpose((2, 0, 1)) # (channel, height, width)
    # turn to float tensor
    screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
    screen = torch.from_numpy(screen)
    return RESIZE(screen).unsqueeze(0).type(Tensor)
    
# def translate_action(actions):
#     return [[('KeyEvent', key[0], key[1]==1) for key in zip(('ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'), actions)]]

if __name__ == "__main__":
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()
    while True:
        action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
        # observation_n: [{'vision':..., 'text':[]}] length of list is number of frame we observed
        # reward_n: [float] length of list is number of frame we observed
        # done_n: [boolean] length of list is number of frame we observed
        # info: {} dictionary of many many information. 
        if(observation_n[-1] != None):
            # A new episode start
            last_screen = get_screen()
            current_screen = get_screen()
            start_flag = True
            done = False
            state = current_screen - last_screen
            
            # Loop for this episode
            for t in count():
                action_flag = select_action(state)
                action = [ACTIONS[action_flag[0][0]] for rw in reward_n]
                
                # Input actions to environment
                _, reward_n, done_n, _  = env.step(action)
                
                reward = sum(reward_n) / 10000
                done = done_n[-1]
                if not done:
                    # Get new observation
                    last_screen = current_screen
                    current_screen = get_screen()
                    next_state = current_screen - last_screen
                else:
                    print("End of one episode")
                    next_state = None
                
                # store to experience pool
                memory.push(state, action_flag, next_state, FloatTensor([reward]).view(1, 1))
                
                # Move to next state
                state = next_state
                
                env.render()
                
                optimize_model()
                if done:
                    episode_duration.append(t+1)
                    plot_durations()
                    break
                
                #print(len(observation_n), reward_n, done_n, info)
        
            # Episode done, save model per 5 episode
            if len(episode_duration) % 5 == 0:
                checkpoint = {"state_dict":model.state_dict(), \
                              "memory":memory, \
                              "steps_done":steps_done, \
                              "episode_duration":episode_duration}
                torch.save(checkpoint, filename)
                print("Episode ", len(episode_duration), " saved to ", filename)
            
            
        env.render()