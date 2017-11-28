'''
Created on Nov 27, 2017

@author: micou, siqi zhang
'''
#==========================================================================
# Description: 
#   This is a modified version of the original DQN implemented by micou.
# Changes:
#   1. Use observation to receive the raw pixel instead of rende
#   2. The input state for deep NN is a image
#   3. Use the universe reward extraction: OCR(optical character recognition) 
#      model that extracts the score
#==========================================================================
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

RESUME = True # When this is True, try to reload last saving point. If got any error in loading, start over and create a new checkpoint
TEST = True # When TEST is true, run the game use the network only, no random to affect the result. Optimizer stop update during this
filename = "./local_training.pth.tar"

# Use model set at DQN.py, model have 4 output
if RESUME:
    try:
        checkpoint = torch.load(filename)
    except:
        print("Error happened in resuming, start from nothing")
        RESUME = False
        
if not RESUME:
    model = DQN()
    if use_cuda:
        model.cuda()
    episode_duration = []
    episode_score = []
    Learning_score = []
    memory = ReplayMemory(10000)
    steps_done = 0
else:
    model = DQN()
    if use_cuda:
        model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    episode_duration = checkpoint['episode_duration']
    try:
        episode_score = checkpoint['episode_score']
    except:
        episode_score = []
    if len(episode_score) < len(episode_duration):
        episode_score = [0 for _ in range(len(episode_duration)-len(episode_score))]+episode_score
    Learning_score = checkpoint['Learning_score']
    memory = checkpoint['memory']
    steps_done = checkpoint['steps_done']
    print("Loaded model, start from episode ", len(episode_duration))
    
# Some global variables
BATCH_SIZE = 250 # used for experience pool based training
GAMMA = 0.99 # Used for expectation decay
EPS_START = 0.95 # using epsilon-greedy algorithm, with this probability to take random step
EPS_END = 0.10 # At the end of training, probability to take random step is quite small
EPS_DECAY = 1000000 # Epsilon decay in exponential speed. Set the decay speed
model_update = 1

reward_change = []
last_sync = 0
optimizer = optim.RMSprop(model.parameters())
RESIZE = T.Compose([T.ToPILImage(),
                    T.Scale(64, interpolation=Image.CUBIC),
                    T.ToTensor()])

def select_action(state):
    '''
    input state and use epsilon-greedy algorithm to select actions to take
    INPUT: state matrix, Tensor
    OUTPUT: [] of 0s and 1s. list length is number of actions. Each state map to one action
    '''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if TEST:
        result = model(Variable(state, volatile=True).type(FloatTensor)).data
        max_index_result = result.max(1)[1].view(1, 1)
        print("Testing... Select ", max_index_result[0][0]," with score ", result.max(1)[0].view(1, 1)[0][0])
        return max_index_result
    if sample > eps_threshold:
        result = model(Variable(state, volatile=True).type(FloatTensor)).data
        max_index_result = result.max(1)[1].view(1, 1)
        print("Select: ", max_index_result[0][0])
        return max_index_result
        
    else:
        rs = random.randrange(ACTION_NUM)
        result = model(Variable(state, volatile=True).type(FloatTensor)).data
        max_result = result.max(1)
        print("Random select ", rs, ", network choose ", max_result[1].view(1, 1)[0][0], " with score ", max_result[0].view(1, 1)[0][0])
        return LongTensor([[rs]])

def plot_durations():
    '''
    Plot training result
    '''
    plt.figure(1)
    plt.clf()
    durations_t = torch.FloatTensor(episode_duration)
    score_t = torch.FloatTensor(episode_score)
    plt.subplot(2, 2, 1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.subplot(2, 2, 2)    
    plt.xlabel('Episode')
    plt.ylabel('TotalReward')
    plt.plot(score_t.numpy())
    if len(score_t) >= 100:
        means = score_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.subplot(2, 2, 3)    
    plt.xlabel('Episode')
    plt.ylabel('Test Result')
    plt.plot(Learning_score)
    plt.subplot(2, 2, 4)
    plt.xlabel("Time")
    plt.ylabel("Reward Change")
    plt.plot([i[0] for i in reward_change], [i[1] for i in reward_change])
    
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
    
    if not TEST:
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
def get_screen():
    screen = env.render(mode="rgb_array")
    # cut out the game image
    screen = screen[84:596, 18:818]
    # turn to float tensor
    screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
    return screen
    
# def translate_action(actions):
#     return [[('KeyEvent', key[0], key[1]==1) for key in zip(('ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'), actions)]]

last_reward= 0
last_time = time.time()
def reward_cal(reward_n, state=None):
    ################################
    # Use log of average speed as a reward, speed_reward
    global last_time
    global last_reward 
    speed_reward = 0
    inlane_reward = 0
    nocrash_reward = 0
    
    for r in reward_n:
        speed_reward += r
    speed_reward = speed_reward * (time.time() - last_time)
    last_time = time.time()
    # Speed reward won't be too large, from 0 to 4+
    speed_reward = np.log(speed_reward+1) 
    ################################
    # Use out of lane detection as a reward, use this block only when state is not None, inlane_reward
    if type(state) in [torch.FloatTensor, torch.LongTensor, torch.ByteTensor, torch.cuda.FloatTensor, torch.cuda.LongTensor, torch.cuda.ByteTensor]:
        if use_cuda:
            img = state[0].cpu().numpy().transpose((1, 2, 0))
        else:
            img = state[0].numpy().transpose((1, 2, 0))
        detect_block = img[420:450, 262:538]
        distance_list = [np.linalg.norm(np.cross(detect_block[i][j]-np.array((0, 0, 0)), detect_block[i][j]-np.array((1, 1, 1))))/np.sqrt(3) for i in range(len(detect_block)) for j in range(len(detect_block[0]))]
        threshold = 15
        print(distance_list)
        percentage_overthreshold = 100*len([i for i in distance_list if i > threshold])/len(distance_list)
        if percentage_overthreshold > 20:
            # if over 20, car get out of lane
            inlane_reward = -2
        else:
            inlane_reward = 0.5
    ################################
    # Use crash detection as a reward, nocrash_reward
    for r in reward_n:
        # Allow normal slow down, like slow down caused by left or right button
        if r>last_reward-50:
            # no punishment
            nocrash_reward = 0.1
        else:
            # Crash happened
            nocrash_reward = -2
        last_reward = r
    
    #final_reward = speed_reward + inlane_reward + nocrash_reward
    final_reward = inlane_reward
    return final_reward 

def transform_nparray2tensor(screen):
    """
    Turn a numpy.array screen matrix to float tensor
    INPUT:
        screen {numpy.ndarray}: 3-channel screen image Height*Width*Channel(3)
    OUTPUT:
        result {FloatTensor}: C*H*W tensor image
    """
    # Transpose the image
    screen = screen.transpose((2, 0, 1)) # (channel, height, width)
    screen = torch.from_numpy(screen)
    result = RESIZE(screen).unsqueeze(0).type(Tensor)
    return result

def get_state(last_screen, current_screen):
    """
    INPUT:
        last_screen {numpy.ndarray}: last screen image
        current_screen {numpy.ndarray}: current_screen image
    OUTPUT:
        state_screen {numpy.ndarray}: state screen. In raw image size
    """
    global steps_done
    # centralize and normalize image 
    cs_mean = current_screen - np.mean(current_screen) # Avoid lighting problem
    # cs_norm = cs_mean/np.std(current_screen) # Because 3-channel already in same scale, no need to do so
    # PCA-Whitening, not used here
#     cov = np.dot(current_screen.T, current_screen)/current_screen.shape[0]
#     U, S, V = np.linalg.svd(cov)
#     cs_rot = np.dot(current_screen, U)
#     cs_white = cs_rot / np.sqrt(S + 1e-5)
    cs_t = cs_mean
    return cs_t

if __name__ == "__main__":
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()
    while True:
        total_reward = 0
        reward_change =[]
        action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
        observation_n, reward_n, done_n, info = env.step(action_n)
        # observation_n: [{'vision':..., 'text':[]}] length of list is number of frame we observed
        # reward_n: [float] length of list is number of frame we observed
        # done_n: [boolean] length of list is number of frame we observed
        # info: {} dictionary of many many information. 
        
        # If observation_n[-1] is still None, environment is not ready
        if(observation_n[-1] != None):
            # A new episode start
            last_screen = get_screen()
            current_screen = get_screen()
            start_flag = True
            done = False
            raw_state = get_state(last_screen, current_screen)
            state = transform_nparray2tensor(raw_state)
            
            # Loop for this episode
            for t in count():
                action_flag = select_action(state)
                action = [ACTIONS[action_flag[0][0]] for rw in reward_n]
                
                # Input actions to environment
                _, reward_n, done_n, _  = env.step(action)
                
                reward = reward_cal(reward_n, raw_state)
                total_reward += sum(reward_n)
                reward_change.append((time.time(), sum(reward_n)))
                done = done_n[-1]
                if not done:
                    # Get new observation
                    last_screen = current_screen
                    current_screen = get_screen()
                    next_raw_state = get_state(last_screen, current_screen)
                    next_state = transform_nparray2tensor(next_raw_state)
                else:
                    print("End of one episode")
                    next_state = None
                
                # store to experience pool
                memory.push(state, action_flag, next_state, FloatTensor([reward]).view(1, 1))
                
                # Move to next state
                state = next_state
                raw_state = next_raw_state
                #plt.imshow(state.cpu().numpy()[-1].transpose(1, 2, 0))
                
                
                env.render()
                
                if t % model_update == 0:
                    optimize_model()
                if done:
                    if TEST:
                        Learning_score.append(total_reward)
                    else:
                        episode_duration.append(t+1)
                        episode_score.append(total_reward)
                    plot_durations()
                    break
                
                #print(len(observation_n), reward_n, done_n, info)
        
            # Episode done, save model per 5 episode
            if len(episode_duration) % 5 == 0:
                checkpoint = {"state_dict":model.state_dict(), \
                              "memory":memory, \
                              "steps_done":steps_done, \
                              "episode_score":episode_score,\
                              "Learning_score":Learning_score,\
                              "episode_duration":episode_duration}
                torch.save(checkpoint, filename)
                print("Episode ", len(episode_duration), " saved to ", filename)
            
            if (len(episode_duration)+len(Learning_score)) % 20 == 0:
                TEST = True
            else:
                TEST = False
            
            
        # If not ready, keep update until ready for training.
        # If finished training done, update and prepare for next episode
        env.render()
