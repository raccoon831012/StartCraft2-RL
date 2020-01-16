#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
#from SC2_PPO import PPO_model
from PPO import PPO_model 
import time
import sys
import sc2gym.envs
from absl import flags
from pysc2.lib import features
import numpy as np
FLAGS = flags.FLAGS
FLAGS(sys.argv)

ITERATION = 10000
EPISODE = 8
BATCH = 4
GAMMA = 0.95

TRAIN = True
#TRAIN = False

if not TRAIN:
    ITERATION = 100
def main():
    #map = 'SC2FindAndDefeatZerglings-v3'
    map = 'SC2CollectMineralShards-v3'
    #map = 'SC2BuildMarines-v1'
    #map ='BuildAndDefeat-v3'
    env = gym.make(map)
    screen_space, mini_space = env.observation_space
    '''
    for iteration in range(ITERATION):  # episode
            screen_obs , mini_obs = env.reset()
            while True:
                act = [-1,-1,-1]
                next_screen_obs,next_mini_obs, reward, done,_ = env.step(act)
                if done:
                    break
    '''
    print(screen_space, mini_space)
    print(env.action_space)
    #screen_space = np.array(screen_space.shape[::-1])
    #mini_space = np.array(mini_space.shape[::-1])

if __name__ == '__main__':
    main()