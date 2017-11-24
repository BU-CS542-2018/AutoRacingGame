import gym
import universe  # register the universe environments
import cv2       # opencv
import numpy
import scipy.misc
import Filters
#Notes:
#     1. use ndarray as mat in python

#constant
IMG_DIM = [768,1024,3]

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1)  # automatically creates a local docker container
observation_n = env.reset()
counter = 0

#now it only takes first 60 frames and save to local folder
while (True and counter < 60):
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
  # some notes
  # 1.observation is a list and the observation is in the [0] laction
  # 2.observation[0] starts off with Nonetype and will become a dictionary once the agent
  #   gets to play the game.
  # 3.the dictionary only has two pair of key-value: text(usually empty) and vision
  if(observation_n[0] != None):
    #image filtering, passing the observation to filter
    grey_out = Filters.obsFilter(observation_n)
    scipy.misc.imsave('frames/outfile'+str(counter)+'.jpg', grey_out)
    counter +=1

  
