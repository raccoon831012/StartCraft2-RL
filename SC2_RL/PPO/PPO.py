import numpy as np
import tensorflow as tf
import gym
# noinspection PyUnresolvedReferences
import sc2gym.envs
from absl import flags
import os
import copy
from network import Network
from network_group import Network_G

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

FLAGS = flags.FLAGS

class PPO_model():
    def __init__(self,obs_space,act_space,sess,network='Network', gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01,learning_rate=1e-4):
        self.sess = sess
        self.obs_space = obs_space
        self.act_space = act_space
        self.learning_rate=learning_rate
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_vs,self.ep_vnt,self.ep_gaes = [], [], [], [],[],[]
        if network =='Network':
            self.Policy = Network('policy',obs_space,act_space,sess)
            self.Old_Policy = Network('old_policy',obs_space,act_space,sess)
        else:
            self.Policy = Network_G('policy',obs_space,act_space,sess)
            self.Old_Policy = Network_G('old_policy',obs_space,act_space,sess)
        self.gamma = gamma
        
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """
        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()
        
        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))
        
        # inputs for train_op
        with tf.variable_scope('train_inp'):
            if network =='Network':
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            else:
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None,len(act_space.nvec)], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_pr = self.Policy.all_act_prob
        act_pr_old = self.Old_Policy.all_act_prob
        log = 0
        if network == 'Network':
            # probabilities of actions which agent took with policy/old policy
            act_probs = act_pr * tf.one_hot(indices=self.actions, depth=act_pr.shape[1])
            act_probs = tf.reduce_sum(act_probs, axis=1)
            act_probs_old = act_pr_old * tf.one_hot(indices=self.actions, depth=act_pr_old.shape[1])
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
            log = tf.log(act_probs) - tf.log(act_probs_old)
        else:
            i =0
            log_probs = 0
            log_probs_old = 0
            for act,act_old in zip(act_pr,act_pr_old):
                act_probs = act * tf.one_hot(indices=self.actions[:,i], depth=act.shape[1])
                log_probs += tf.log(tf.reduce_sum(act_probs, axis=1))
                act_probs_old = act_old * tf.one_hot(indices=self.actions[:,i], depth=act_old.shape[1])
                log_probs_old += tf.log(tf.reduce_sum(act_probs_old, axis=1))
                i=i+1
            
            log = log_probs - log_probs_old

        with tf.variable_scope('loss/clip'): #paper (7)
            ratios = tf.exp(log)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

        # construct computation graph for loss of value function
        with tf.variable_scope('loss/vf'): 
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('loss_vf', loss_vf)
        
        # construct computation graph for loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            entropy = 0
            if network == 'Network':
                entropy_ = -tf.reduce_sum(self.Policy.all_act_prob *
                                        tf.log(tf.clip_by_value(self.Policy.all_act_prob, 1e-10, 1.0)), axis=1)
                entropy = tf.reduce_mean(entropy_, axis=0)  # mean of entropy of pi(obs)
            else:
                for act in self.Policy.all_act_prob:
                    entropy_ = -tf.reduce_sum(act *
                                        tf.log(tf.clip_by_value(act, 1e-10, 1.0)), axis=1)
                    entropy += tf.reduce_mean(entropy_, axis=0)  # mean of entropy of pi(obs)

            tf.summary.scalar('entropy', entropy)

        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy #paper (9)
            alpha = 0.9
            beta = 0.2
            gamma_ = 1
            #loss = alpha * loss_clip + beta * entropy - gamma_ * loss_vf
            loss = -loss  # minimize -loss == maximize loss
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list=pi_trainable)
    
    def train(self):
        obs, actions, rewards, v_preds_next, gaes = self.array_reshape()
        self.sess.run(self.train_op, feed_dict={self.Policy.tf_obs: obs,
                                                                 self.Old_Policy.tf_obs: obs,
                                                                 self.actions: actions,
                                                                 self.rewards: rewards,
                                                                 self.v_preds_next: v_preds_next,
                                                                 self.gaes: gaes})

    def get_summary(self):
        obs, actions, rewards, v_preds_next, gaes = self.array_reshape()
        return self.sess.run(self.merged, feed_dict={self.Policy.tf_obs: obs,
                                                                      self.Old_Policy.tf_obs: obs,
                                                                      self.actions: actions,
                                                                      self.rewards: rewards,
                                                                      self.v_preds_next: v_preds_next,
                                                                      self.gaes: gaes})
    def clear_all_array(self):
        self.ep_obs, self.ep_as, self.ep_rs, self.ep_vs,self.ep_vnt,self.ep_gaes = [], [], [], [],[],[]
    
    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)
    def choose_action(self,obs):
        obs_ = np.reshape(obs, newshape=[-1,self.obs_space[0],self.obs_space[1],self.obs_space[2]])
        return self.Policy.choose_action(obs_)

    def get_gaes(self):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(self.ep_rs, self.ep_vnt, self.ep_vs)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        ga = np.array(gaes)
        ga = (ga - ga.mean()) / ga.std()
        return ga
    
    def store_transition(self, s, a, r, v):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        self.ep_vs.append(v)

    def array_reshape(self):
        obs = np.reshape(self.ep_obs, newshape=[-1,self.obs_space[0],self.obs_space[1],self.obs_space[2]])
        actions = np.array(self.ep_as).astype(dtype=np.int32)
        rewards = np.array(self.ep_rs).astype(dtype=np.float32)
        v_preds_next = np.array(self.ep_vs).astype(dtype=np.float32)
        gaes = np.array(self.ep_gaes).astype(dtype=np.float32)
        return obs, actions, rewards, v_preds_next, gaes

    def save_trainer(self,fname):
        saver = tf.train.Saver()
        dirname = os.path.dirname(fname)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        save_path = saver.save(self.sess, fname)
        print("Model saved in path: %s" % save_path)
    
    def restore_trainer(self,fname):
        f = os.path.dirname(fname)+'/checkpoint'
        #print(os.path.isfile(f),f)
        if os.path.isfile(f):
            saver = tf.train.Saver()
            saver.restore(self.sess, fname)
            print("Model restored in path")
        else:
            print("Not found Model from", fname)