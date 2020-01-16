#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
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

#TRAIN = True
TRAIN = False

if not TRAIN:
    ITERATION = 1
def main():
    #map = 'SC2MoveToBeacon-v1'
    #map = 'SC2CollectMineralShards-v2'
    #map = 'SC2FindAndDefeatZerglings-v2'
    #map = 'SC2FindAndDefeatZerglings-v3'
    #map = 'SC2DefeatZerglingsAndBanelings-v1'
    #map = 'SC2DefeatZerglingsAndBanelings-v2'
    #map = 'SC2DefeatRoaches-v1'
    #map = 'SC2DefeatRoaches-v2'
    map = 'SC2CollectMineralsAndGas-v0'
    #map = 'SC2BuildMarines-v0'
    env = gym.make(map)
    '''
    for iteration in range(ITERATION):  # episode
            obs = env.reset()
            while True:
                act = [-1,-1,-1]
                next_obs, reward, done,_ = env.step(act)
                if done:
                    break
    '''
    screen_space = env.observation_space
    print(screen_space)
    screen_space = np.array(screen_space.shape[::-1])

if __name__ == '__main__':
    main()