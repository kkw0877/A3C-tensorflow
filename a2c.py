import numpy as np
import tensorflow as tf

import random
from collections import namedtuple

import model
import model_helper
from utils import misc_utils

class A2C(model.PolicyGradient):
    def build_graph(self, flags):
        """Build A2C graph."""
        with tf.variable_scope(flags.model_name):
            # TODO. Fix tf.device for multiple gpu
            with tf.device(model_helper.get_device_str(0, self.num_gpus)):
                c1 = self.conv2d(self.state, self.cv_num_outputs, 
                                 self.f_height, self.f_width, 
                                 self.stride, scope="conv2d_1")
                c2 = self.conv2d(c1, self.cv_num_outputs*2, 
                                 self.f_height/2, self.f_width/2, 
                                 self.stride/2, scope="conv2d_2")
                fc = self.linear(self.flatten(c2), self.num_units,
                                 activation_fn=tf.nn.relu, scope='flat')

                # modify the shape of the fc before rnn
                # [1, None, self.flat_outputs]
                rnn_input = tf.reshape(fc, [1, -1, self.num_units])     
                step_size = tf.shape(rnn_input)[1:2] 

                cell = self.create_rnn_cell()
                self.h_in = cell.zero_state(1, tf.float32)

                rnn_output, self.h_out = tf.nn.dynamic_rnn(
                    cell, rnn_input, initial_state=self.h_in, 
                    sequence_length=step_size)  
                rnn_output = tf.reshape(rnn_output, [-1, self.num_units])

                # policy
                self.policy = self.linear(
                    rnn_output, self.action_size,
                    activation_fn=tf.nn.softmax, scope='policy')
                # value
                self.value = self.linear(
                    rnn_output, 1, scope='value')

                # compute loss
                if self.mode != tf.estimator.ModeKeys.PREDICT:
                    loss = self.compute_loss()
                else:
                    loss = tf.constant(0.0)

                return loss
        
    def compute_loss(self):
        """Compute Optimization loss"""
        policy = self.policy
        value = self.value
        
        self.action = tf.placeholder(tf.int32, [None])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.target = tf.placeholder(tf.float32, [None])
        
        # actor loss
        actions_onehot = tf.one_hot(self.action, self.action_size)
        action_probability = tf.reduce_sum(policy * actions_onehot, axis=1)
        actor_loss = -tf.reduce_sum(tf.log(action_probability+1e-10) * self.advantage)
        
        # critic loss
        value = tf.reshape(value, [-1])
        critic_loss = tf.reduce_sum(tf.square(self.target - value))
        
        # entropy loss for exploration
        entropy_loss = tf.reduce_sum(policy * tf.log(policy+1e-10), axis=1)
        entropy_loss = tf.reduce_sum(entropy_loss)
        
        total_loss = actor_loss + critic_loss + entropy_loss*0.01
        return total_loss
        
    def get_action(self, sess, state, hidden):
        """Get action through the network"""
        policy, value, next_hidden = sess.run([self.policy, self.value, self.h_out], 
            feed_dict={self.state:[state], self.h_in:hidden})
        
        policy = policy[0] 
        value = value[0, 0] 
        action = np.random.choice(self.action_size, p=policy)
        return action, policy, value, next_hidden
    
    def update_network(self, sess, state, hidden, next_state,
                       next_hidden, action, value, reward, episode_end):
        """Update network."""
        next_value = 0.
        if not episode_end:
            next_value = sess.run(self.value,
                feed_dict={self.state:[next_state],
                           self.h_in:next_hidden})[0, 0]
        
        next_value = reward + self.discount_factor * next_value
        advantage = next_value - value
        
        sess.run(self.update_op, feed_dict={self.state:[state], 
                                            self.h_in:hidden,
                                            self.action:[action], 
                                            self.target:[next_value], 
                                            self.advantage:[advantage]}) 
            
    def train(self, sess):
        '''Execute train graph.'''
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        
        # episode_end, start_life, dead, state, hidden, stats 
        episode_vars = self.before_episode(sess)

        state = episode_vars.state
        hidden = episode_vars.hidden
        start_life = episode_vars.start_life

        episode_end = episode_vars.episode_end
        dead = episode_vars.dead
        stats = episode_vars.stats
        
        # set initial_h for the update
        initial_h = hidden        
        
        while not episode_end:
            # get action, policy, value and next_hidden through the network
            action, policy, value, next_hidden = self.get_action(sess, state, hidden)
            
            # 1. stop, 2. left, 3. right
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3
                
            if dead:
                action = 0
                real_action = 1
                dead = False
            
            # move 
            next_state, reward, episode_end, info = self.env.step(real_action)
            next_state = misc_utils.preprocess_image(next_state, self.img_height, self.img_width)
            
            # update stats about duration, avg_max_q, total_reward
            self.update_stats(stats, reward, policy)
            
            # update network for every time step
            self.update_network(sess, state, hidden, next_state,
                                next_hidden, action, value, reward, episode_end)
            
            # if dead
            if start_life > info['ale.lives']:
                start_life = info['ale.lives']
                dead = True
                
            # change state to next_state
            state = next_state
            # change hidden to next_hidden
            hidden = next_hidden

        print("Episode done. avg_max_q: {:.4f}, total_reward: {}".format(
            stats['avg_max_q']/stats['episode_step'], stats['total_reward']))

        # When the episode ends, update network and summary_vars 
        # then, return the train result
        self.update_summary_vars(sess, stats)
        output_tuple = model.TrainOutput(global_step=self.global_step, train_summary=self.summary)
        return sess.run(output_tuple)    
        
    def infer(self, sess):
        """Execute infer graph."""
        assert self.mode == tf.estimator.ModeKeys.PREDICT
        
        # episode_end, start_life, dead, state, hidden, stats 
        episode_vars = self.before_episode(sess)
        
        state = episode_vars.state
        hidden = episode_vars.hidden
        start_life = episode_vars.start_life

        episode_end = episode_vars.episode_end
        dead = episode_vars.dead
        stats = episode_vars.stats   
            
        while not episode_end:
            # get action, policy, value and next_hidden through the network
            action, policy, value, next_hidden = self.get_action(sess, state, hidden)
            
            # 1. stop, 2. left, 3. right
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            if dead:
                action = 0
                real_action = 1
                dead = False

            # visualize
            self.env.render()
                    
            # move 
            next_state, reward, episode_end, info = self.env.step(real_action)
            next_state = misc_utils.preprocess_image(next_state, self.img_height, self.img_width)
                
            # update stats about duration, avg_max_q, total_reward
            self.update_stats(stats, reward, policy)
                
            # if dead
            if start_life > info['ale.lives']:
                start_life = info['ale.lives']
                dead = True
                  
            # change state to next_state
            state = next_state
            # change hidden to next_hidden
            hidden = next_hidden
            
        # end episode
        print("Infer_mode done. avg_max_q: {:.4f}, total_reward: {}".format(
            stats['avg_max_q']/stats['episode_step'], stats['total_reward']))
        self.update_summary_vars(sess, stats)
        infer_output = model.InferOutput(infer_summary=self.summary)
        
        return sess.run(infer_output)
