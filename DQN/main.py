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

RESUME = True # When this is True, try to reload last saving point. If got any error in loading, start over and create a new checkpoint
TEST = False # When TEST is true, run the game use the network only, no random to affect the result. Optimizer stop update during this
filename = "./local_training.pth.tar"
best_filename = "./currently_best.pth.tar"

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
    current_best_test_reward = 0
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
    try:
        current_best_test_reward = checkpoint["current_best_test_reward"]
    except:
        current_best_test_reward = 0
    if len(episode_score) < len(episode_duration):
        episode_score = [0 for _ in range(len(episode_duration)-len(episode_score))]+episode_score
    Learning_score = checkpoint['Learning_score']
    memory = checkpoint['memory']
    steps_done = checkpoint['steps_done']
    print("Loaded model, start from episode ", len(episode_duration))
    print("Current Best test result is ", current_best_test_reward)
    time.sleep(5)
    
# Some global variables
BATCH_SIZE = 250 # used for experience pool based training
GAMMA = 0.99 # Used for expectation decay
EPS_START = 0.95 # using epsilon-greedy algorithm, with this probability to take random step
EPS_END = 0.10 # At the end of training, probability to take random step is quite small
EPS_DECAY = 1000000 # Epsilon decay in exponential speed. Set the decay speed
punishment = -3
model_update = 1

# Global but only used in episode variables. 
reward_change = []
final_rewards= []
last_frames = []
last_frames_to_save = 4
last_reward= 0
last_sync = 0
last_time = time.time()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
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
    global final_rewards
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
    real_rewards = [i[1] for i in reward_change]
    max_read_rewards = max(real_rewards)
    min_read_rewards = min(real_rewards)
    read_reards_length = max_read_rewards-min_read_rewards+1
    if not TEST:
        designed_rewards = [i[1] for i in final_rewards]
        max_designed_rewards = max(designed_rewards)
        min_designed_rewards = min(designed_rewards)
        designed_rewards_length = max_designed_rewards-min_designed_rewards+1
        plt.plot([i[0] for i in reward_change], [i/read_reards_length for i in real_rewards], "b", \
                 [i[0] for i in final_rewards], [i/designed_rewards_length for i in designed_rewards], "r")
    else:
        plt.plot([i[0] for i in reward_change], [i/read_reards_length for i in real_rewards], "b")
    plt.pause(0.001)

def optimize_model():
    """ERROR"""
    global last_sync
    global punishment
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
    # screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
    return screen
    
# def translate_action(actions):
#     return [[('KeyEvent', key[0], key[1]==1) for key in zip(('ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'), actions)]]

def reward_cal(reward_n, state=None):
    ################################
    # Use log of average speed as a reward, speed_reward
    global last_time
    global last_reward 
    global punishment
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
    if type(state) in [torch.FloatTensor, torch.LongTensor, torch.ByteTensor]:
        img = state[0].numpy().transpose((1, 2, 0))
    elif type(state) in [torch.cuda.FloatTensor, torch.cuda.LongTensor, torch.cuda.ByteTensor]:
        img = state[0].cpu().numpy().transpose((1, 2, 0))
    elif type(state) in [np.ndarray]:
        img = state
    else:
        img = 0
        
    if not isinstance(img, int):
        detect_block = img[420:450, 262:538]
        threshold = 15/255
        distance_list = (np.linalg.norm(np.cross(detect_block-np.array((0, 0, 0)), detect_block-np.array((1, 1, 1))), axis=2)/np.sqrt(3)).flatten()
        percentage_overthreshold = 100*len(np.extract(distance_list>threshold, distance_list))/len(distance_list)
        if percentage_overthreshold > 20:
            # if over 20, car get out of lane
            inlane_reward = punishment/2
        else:
            inlane_reward = 0
    ################################
    # Use crash detection as a reward, nocrash_reward
    for r in reward_n:
        # Allow normal slow down, like slow down caused by left or right button
        if r>last_reward-50:
            # no punishment
            nocrash_reward = 0
        else:
            # Crash happened
            nocrash_reward = punishment/2
        last_reward = r
    #print(speed_reward, inlane_reward, nocrash_reward, "current punishment: ", punishment)
    final_reward = speed_reward + inlane_reward + nocrash_reward
    #final_reward = inlane_reward
    return final_reward 

def transform_nparray2tensor(screen):
    """
    Turn a numpy.array screen matrix to float tensor
    INPUT:
        screen {numpy.ndarray}: 3-channel screen image Height*Width*Channel(3)
    OUTPUT:
        result {FloatTensor}: C*H*W tensor image
    """
    # resize screen
    screen = cv2.resize(screen, (100, 64), interpolation=cv2.INTER_LINEAR)
    # turn to float tensor
    screen = np.ascontiguousarray(screen, dtype = np.float32) / 255
    # Transpose the image
    screen = screen.transpose((2, 0, 1)) # (channel, height, width)
    result = torch.from_numpy(screen)
    # Turn 3D tensor to 4D tensor (NCHW, number of inputs, channels, height, width)
    result=result.unsqueeze(0).type(Tensor)
#     result = RESIZE(screen).unsqueeze(0).type(Tensor)
    return result

def get_state(last_screen, current_screen, observation_n=[]):
    """
    INPUT:
        last_screen {numpy.ndarray}: last screen image
        current_screen {numpy.ndarray}: current_screen image
        observation_n {[{'vision':..., 'text':...}, ...]}: (Optional) observations
    OUTPUT:
        state_screen {numpy.ndarray}: state screen. In raw image size
    """
    global steps_done
    global last_frames_to_save
    global last_frames
    # centralize and normalize image 
    cs_mean = current_screen - np.mean(current_screen) # Avoid lighting problem
    # Get last few frames
    if len(observation_n) >= last_frames_to_save-1:
        last_frames = [ob['vision'][84:596, 18:818] for ob in observation_n[(-last_frames_to_save+1):]]+[current_screen]
    else:
        last_frames = last_frames[-(last_frames_to_save-1-len(observation_n)):]+[ob['vision'][84:596, 18:818] for ob in observation_n]+[current_screen]
    # Get road segmentation
    road_segmentation=np.linalg.norm(np.cross(current_screen-np.array((0, 0, 0)), current_screen-np.array((1, 1, 1))), axis=2)/np.sqrt(3)
    _, road_segmentation=cv2.threshold(road_segmentation, 15, 255, cv2.THRESH_BINARY)
    road_segmentation=cv2.dilate(road_segmentation, np.ones((2, 2), np.uint8), iterations=3)[..., None]
    # Get canny channels
    cys = [cv2.Canny(frame, 100, 180, L2gradient=True)[..., None] for frame in last_frames]
    # Put them together
    cs_t = np.concatenate([road_segmentation, cs_mean]+cys, axis=2)
    return cs_t

if __name__ == "__main__":
    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1)  # automatically creates a local docker container
    observation_n = env.reset()
    
    while True:
        total_reward = 0
        reward_change =[]
        final_rewards = []
        last_frames = []
        
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
            last_frames = [np.zeros(current_screen.shape, dtype=np.uint8) for _ in range(last_frames_to_save) ]
            raw_state = get_state(last_screen, current_screen)
            state = transform_nparray2tensor(raw_state)
            
            # Loop for this episode
            for t in count():
                action_flag = select_action(state)
                action = [ACTIONS[action_flag[0][0]] for rw in reward_n]
                
                # Input actions to environment
                observation_n, reward_n, done_n, _  = env.step(action)
                
                reward = reward_cal(reward_n, current_screen)
                total_reward += sum(reward_n)
                current_time= time.time()
                reward_change.append((current_time, sum(reward_n)))
                final_rewards.append((current_time, reward))
                done = done_n[-1]
                if not done:
                    # Get new observation
                    last_screen = current_screen
                    current_screen = get_screen()
                    next_raw_state = get_state(last_screen, current_screen, observation_n)
                    next_state = transform_nparray2tensor(next_raw_state)
                else:
                    print("End of one episode")
                    next_state = None
                
                # store to experience pool
                if not TEST:
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
                        if total_reward > current_best_test_reward:
                            current_best_test_reward = total_reward
                            checkpoint = {"state_dict":model.state_dict(), \
                                          "memory":memory, \
                                          "steps_done":steps_done, \
                                          "episode_score":episode_score,\
                                          "Learning_score":Learning_score,\
                                          "current_best_test_reward":current_best_test_reward, \
                                          "episode_duration":episode_duration}
                            torch.save(checkpoint, best_filename)
                            print("Episode ", len(episode_duration), " saved to ", best_filename)
                        TEST=False
                    else:
                        episode_duration.append(t+1)
                        episode_score.append(total_reward)
                        # Update reward? Testing
#                         if punishment > -10 and punishment < 10:
#                             punishment-=sum([i[1] for i in final_rewards] )/len(final_rewards)
#                         print("New punishement: ", punishment)
                        TEST= False
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
                              "current_best_test_reward":current_best_test_reward, \
                              "episode_duration":episode_duration}
                torch.save(checkpoint, filename)
                print("Episode ", len(episode_duration), " saved to ", filename)
            
            # Run a test use data generated by network only every 20 rounds
            if (len(episode_duration)+len(Learning_score)) % 20 == 0:
                TEST = True
            else:
                TEST = False
            
            
        # If not ready, keep update until ready for training.
        # If finished training done, update and prepare for next episode
        env.render()