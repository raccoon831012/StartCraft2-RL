import numpy as np
import tensorflow as tf
import gym
# noinspection PyUnresolvedReferences
import sc2gym.envs
from absl import flags
import os

FLAGS = flags.FLAGS

class Network():
    def __init__(self, name: str, obs_space, act_space,sess, temp=0.1,):
        self.sess = sess
        """
        :param name: string
        :param env: gym env
        :param temp: temperature of boltzmann distribution
        """
        #self.n_features = obs
        self.obs_space = obs_space
        self.act_space = act_space
        self.buildnet(name)
    
    def buildnet(self,name):
        with tf.variable_scope(name):
            self.tf_obs = tf.placeholder(dtype=tf.float32, shape = [None, self.obs_space[0],self.obs_space[1],self.obs_space[2]], name='obs')
            #self.obs_mini = tf.placeholder(dtype=tf.float32, shape = [None, self.obs_space[0],self.obs_space[1],self.obs_space[2]], name='obs')
            with tf.variable_scope('policy_net'):
                self.CNN_output = self.CNN()
                layer = tf.layers.dense(
                    inputs=self.CNN_output,
                    units=100,
                    activation=tf.nn.tanh,  # tanh activation
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='fc1'
                )
                layer2 = tf.layers.dense(
                    inputs=layer,
                    units=100,
                    activation=tf.nn.tanh,  # tanh activation
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer2'
                )
                layer3 = tf.layers.dense(
                    inputs=layer2,
                    units=100,
                    activation=tf.nn.tanh,  # tanh activation
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer3'
                )
                layer4 = tf.layers.dense(
                    inputs=layer3,
                    units=100,
                    activation=tf.nn.tanh,  # tanh activation
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='layer4'
                )
                self.all_act_prob = self.outlayer(layer4)
            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.CNN_output, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)
            self.scope = tf.get_variable_scope().name
    def outlayer(self,layer):
        all_act_prob = tf.layers.dense(
                    inputs=layer,
                    units=self.act_space.n,
                    activation=tf.nn.softmax,
                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='fc2'
                )
        return all_act_prob
    def choose_action(self,obs):
        prob_weights,v_pred = self.sess.run([self.all_act_prob,self.v_preds], feed_dict={self.tf_obs: obs})
        #print(v_pred.flatten()[0])
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action,v_pred.flatten()[0]
    
    def CNN(self):
        conv1 = tf.layers.conv2d(
                    inputs=self.tf_obs,
                    filters=32,
                    strides=[4,4],
                    kernel_size=[9, 9],
                    padding="same",
                    activation=tf.nn.relu
                )
        h1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv2 = tf.layers.conv2d(
                    inputs=h1,
                    filters=16,
                    strides=[2,2],
                    kernel_size=[9, 9],
                    padding="same",
                    activation=tf.nn.relu
                )
        h2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        flat = tf.layers.Flatten()(h2)
        return flat

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)