import numpy as np
import tensorflow as tf
import gym
# noinspection PyUnresolvedReferences
import sc2gym.envs
from absl import flags
import os
from network import Network
FLAGS = flags.FLAGS

class Network_G(Network):
    def outlayer(self,layer):
        all_act_prob = []
        count = 0
        for i in self.act_space.nvec:
            act_prob = tf.layers.dense(
                        inputs=layer,
                        units=i,
                        activation=tf.nn.softmax,
                        name="output"+str(count)
                    )
            count += 1
            all_act_prob.append(act_prob)
        #print(all_act_prob)
        return all_act_prob

    def choose_action(self,obs):
        prob_weights,v_pred = self.sess.run([self.all_act_prob,self.v_preds], feed_dict={self.tf_obs: obs})
        #print(v_pred.flatten()[0])
        action = []
        for elemnet in prob_weights:
            choice = np.random.choice(range(elemnet.shape[1]), p=elemnet.ravel())
            action.append(choice)
        #print(prob_weights)
        return action,v_pred.flatten()[0]
    
