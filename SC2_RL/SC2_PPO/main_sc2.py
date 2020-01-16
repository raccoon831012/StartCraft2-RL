#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
from SC2_PPO import PPO_model
import time
import sys
import sc2gym.envs
from absl import flags
from pysc2.lib import features
import numpy as np
FLAGS = flags.FLAGS
FLAGS(sys.argv)

ITERATION = 10000
EPISODE = 4#8
BATCH = 2#4
GAMMA = 0.95

#TRAIN = True
TRAIN = False

if not TRAIN:
    ITERATION = 100
def main():
    #map = 'SC2MoveToBeacon-v3'
    map = 'SC2CollectMineralShards-v3'
    #map = 'SC2FindAndDefeatZerglings-v2'
    #map = 'SC2FindAndDefeatZerglings-v3'
    #map = 'SC2DefeatZerglingsAndBanelings-v1'
    #map = 'SC2DefeatZerglingsAndBanelings-v2'
    #map = 'SC2DefeatRoaches-v1'
    #map = 'SC2DefeatRoaches-v2'
    #map = 'SC2BuildMarines-v0'
    #map = 'SC2BuildMarines-v1'
    #map = 'SC2BuildMarinesBigMap-v1'
    #map = 'BuildAndDefeat-v3'

    env = gym.make(map)
    screen_space, mini_space = env.observation_space
    screen_space = np.array(screen_space.shape[::-1])
    mini_space = np.array(mini_space.shape[::-1])
    #print(screen_space)
    #print(mini_space)
    #act_space = np.prod(env.action_space.nvec)
    act_space = env.action_space
    with tf.Session() as sess:
        #print('act_space',act_space.nvec[0])
        net = 'network_G'
        fold = "one/"+map
        if net == 'network_G':
            fold = "group/"+map
            #fold = "mine/"+map
        PPO = PPO_model(screen_space,mini_space,act_space,sess,network=net, gamma=GAMMA)
        sess.run(tf.global_variables_initializer())
        PPO.restore_trainer("./ckpt/"+fold+"/summary/model.ckpt")
        total_reward=0
        max_reward=0
        rewards=0
        if TRAIN:
            writer = tf.summary.FileWriter("./log/"+fold, sess.graph)
        for iteration in range(ITERATION):  # episode
            screen_obs , mini_obs = env.reset()
            while True:
                act, v_pred = PPO.choose_action(screen_obs , mini_obs)
                next_screen_obs,next_mini_obs, reward, done,_ = env.step(act)
                if max_reward < reward:
                    max_reward=reward
                total_reward+=reward
                rewards+=reward
                if TRAIN:
                    if done and iteration % BATCH==0:
                        #不儲存最後一個
                        PPO.ep_vnt = PPO.ep_vs[1:]+[v_pred]
                        PPO.ep_gaes = PPO.get_gaes()
                        PPO.assign_policy_parameters()
                        for _ in range(EPISODE):
                            PPO.train()
                        summary = PPO.get_summary()
                        writer.add_summary(summary, iteration)
                        PPO.clear_all_array()
                        break
                    elif done:
                        break
                    else:
                        PPO.store_transition(screen_obs , mini_obs,act,reward*100,v_pred)
                elif done:
                    break
                else:
                    #time.sleep(0.1)
                    pass

                screen_obs , mini_obs = next_screen_obs,next_mini_obs
            if max_reward < rewards:
                    max_reward = rewards
            rewards = 0
            if TRAIN:
                if iteration % 4 == 0:
                    PPO.save_trainer("./ckpt/"+fold+"/model.ckpt")
                tfsum = tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(PPO.ep_rs)/100)])
                writer.add_summary(tfsum, iteration)
        print("average reward:",total_reward/ITERATION)
        print("max reward:",max_reward)
        if TRAIN:
            writer.close()
            PPO.save_trainer("./ckpt/"+fold+"/summary/model.ckpt")
    sess.close()



if __name__ == '__main__':
    #maps=['SC2MoveToBeacon-v1','SC2CollectMineralShards-v2','SC2FindAndDefeatZerglings-v2','SC2DefeatZerglingsAndBanelings-v1','SC2DefeatRoaches-v1']
    #map = 'SC2CollectMineralShards-v2'
    main()