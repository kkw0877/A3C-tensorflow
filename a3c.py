import gym
import numpy as np
import tensorflow as tf

import time
import random
import threading
from collections import namedtuple

import a2c
import model
import model_helper
from utils import misc_utils


class A3C(a2c.A2C):
    def __init__(self, flags, mode):
        super(A3C, self).__init__(flags=flags, mode=mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.make_coord_and_agents(flags)

    def make_coord_and_agents(self, flags):
        """Make local agents and coordinator."""
        print("# Making %d agents" % flags.thread_num)
        self.agents = [Agent(flags,
                             state=self.state,
                             h_in=self.h_in,
                             action=self.action,
                             advantage=self.advantage,
                             target=self.target,
                             summary_tags=self.summary_tags,
                             summary_placeholders=self.summary_placeholders,
                             summary_assign_ops=self.summary_assign_ops,
                             update_op=self.update_op,
                             discount_factor=self.discount_factor,
                             thread_num=thread_num) for thread_num in range(flags.thread_num)]

        self.coord = tf.train.Coordinator()
        
    def train(self, sess):
        """Execute train graph through multi thread."""
        agent_threads = []
        for agent in self.agents:
            thread = threading.Thread(target=agent.run, args=(sess,))
            thread.start()
            time.sleep(1)
            agent_threads.append(thread)
  
        self.coord.join(agent_threads)
        output_tuple = model.TrainOutput(global_step=self.global_step, train_summary=self.summary)
        return sess.run(output_tuple)

class Agent(object):
    def __init__(self, flags, **kwargs):
        self.set_hparams_init(flags, kwargs)
        self.build_graph(kwargs) 
        self.update_local_network(flags.model_name, self.thread_var_scope) # from_scope, to_scope
    
    def set_hparams_init(self, flags, kwargs):
        # Set vars in order to update global network
        self.state = kwargs['state']
        self.h_in = kwargs['h_in']
        self.action = kwargs['action']
        self.advantage = kwargs['advantage']
        self.target = kwargs['target']
        
        self.discount_factor = kwargs['discount_factor']
        self.update_op = kwargs['update_op']

        self.thread_var_scope = "%d_thread" % kwargs['thread_num']
        
        # Set vars to update the performance of the global network
        self.summary_tags = kwargs['summary_tags']
        self.summary_placeholders = kwargs['summary_placeholders']
        self.summary_assign_ops = kwargs['summary_assign_ops']
        
        # Set environment 
        self.env = gym.make(flags.env)
        self.local_state = tf.placeholder(
            tf.float32, [None, flags.img_height, flags.img_width, 1])
        self.action_size = 3
        
        # Set image height, width
        self.img_height = flags.img_height
        self.img_width = flags.img_width
        
        self.num_gpus = flags.num_gpus
        
        # Initializer for weights, biases
        self.random_seed = flags.random_seed
        self.w_init = model_helper.get_initializer(
            flags.w_init_op, self.random_seed, flags.mean, flags.stddev)
        self.b_init = model_helper.get_initializer(
            flags.b_init_op, self.random_seed, bias_start=flags.bias_start)
        
        # Convolution
        self.cv_num_outputs = flags.cv_num_outputs
        self.f_height = flags.f_height # filter height
        self.f_width = flags.f_width # filter width
        self.stride = flags.stride
        self.padding = flags.padding
        
        # Recurrent
        self.rnn_num_layers = flags.rnn_num_layers
        self.cell_type = flags.cell_type
        self.num_units = flags.num_units
        self.dropout = flags.dropout
        self.residual_connect = flags.residual_connect
        
        # Set var related to the period of update
        self.t_max = flags.t_max
        
    def build_graph(self, kwargs):
        with tf.variable_scope(self.thread_var_scope):
            # TODO. will fix tf.device for multiple gpu
            with tf.device(model_helper.get_device_str(0, self.num_gpus)):
                c1 = self.conv2d(self.local_state, self.cv_num_outputs, 
                                 self.f_height, self.f_width,
                                 self.stride, scope="conv2d_1") 
                c2 = self.conv2d(c1, self.cv_num_outputs*2, 
                                 self.f_height/2, self.f_width/2,
                                 self.stride/2, scope="conv2d_2")
                fc = self.linear(self.flatten(c2), self.num_units, 
                                 activation_fn=tf.nn.relu, scope="flat")

                # modify the shape of the fc before rnn
                # [1, None, self.flat_outputs]
                rnn_input = tf.reshape(fc, [1, -1, self.num_units]) 
                step_size = tf.shape(rnn_input)[1:2] 

                cell = self.create_rnn_cell()
                self.local_h_in = cell.zero_state(1, tf.float32)

                rnn_output, self.h_out = tf.nn.dynamic_rnn(
                    cell, rnn_input, initial_state=self.local_h_in, 
                    sequence_length=step_size)
                rnn_output = tf.reshape(rnn_output, [-1, self.num_units])

                # policy
                self.policy = self.linear(
                    rnn_output, self.action_size,
                    activation_fn=tf.nn.softmax, scope="policy")
                # value
                self.value = self.linear(
                    rnn_output, 1, scope="value")
    
    def update_local_network(self, from_scope, to_scope): 
        """the vars of the local network is copied from the vars of the global network"""
        # global network vars
        from_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, from_scope) 
        # local network vars
        to_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, to_scope) 
        
        self.update_ops_holder = [to_var.assign(from_var) for from_var, to_var 
                                  in zip(from_vars, to_vars)]
        
    def get_action(self, sess, state, hidden):
        """Get action through the network"""
        policy, value, next_hidden = sess.run([self.policy, self.value, self.h_out], 
            feed_dict={self.local_state:[state], self.local_h_in:hidden})
        
        policy = policy[0] 
        value = value[0, 0] 
        action = np.random.choice(self.action_size, p=policy)
        
        return action, policy, value, next_hidden 
        
    def discount_rewards(self, sess, rewards, episode_end, state, hidden):
        """Get discount rewards to update the network."""
        value = 0.
        discounted_prediction = np.zeros_like(rewards)
        
        # If an episode ends, the value of the state is zero.
        # Otherwise, get the value of the state through the network.
        if not episode_end:
            value = sess.run(self.value, 
                feed_dict={self.local_state:[state], 
                           self.local_h_in:hidden})[0, 0]
            
        for i in reversed(range(len(rewards))):
            value = rewards[i] + self.discount_factor * value
            discounted_prediction[i] = value
            
        return discounted_prediction
    
    def append_sample(self, s, a, r, v):
        # state, action, reward, value
        self.s_list.append(s)
        self.a_list.append(a)
        self.r_list.append(r)
        self.v_list.append(v)
    
    def list_reset(self):
        """reset s_list, a_list, r_list for train mode."""
        self.s_list = [] # state
        self.a_list = [] # action
        self.r_list = [] # reward
        self.v_list = [] # value
    
    def before_episode(self, sess):
        """ initialize vars related to the Atari environment before start."""
        self.list_reset()
        
        state = self.env.reset()
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = self.env.step(1)
        state = misc_utils.preprocess_image(state, self.img_height, self.img_width)
        hidden = sess.run(self.local_h_in)
        
        # start_life, episode_end, dead, cur_time
        start_life = 5
        episode_end = False
        dead = False
        cur_time = 0 # only use for A3C
        
        # the model performance
        stats = {tag: 0.0 for tag in self.summary_tags}
        
        # set episode_vars related to the episode
        episode_vars = model.EpisodeVars(state=state,
                                         hidden=hidden,
                                         start_life=start_life,
                                         episode_end=episode_end,
                                         dead=dead,
                                         cur_time=cur_time,
                                         stats=stats)
        
        return episode_vars
    
    def update_stats(self, stats, reward, policy):
        """Update stats"""
        stats['episode_step'] += 1.0
        stats['total_reward'] += reward
        stats['avg_max_q'] += np.amax(policy)

    def run(self, sess):
        # episode_end, start_life, dead, state, hidden, stats 
        episode_vars = self.before_episode(sess)

        state = episode_vars.state
        hidden = episode_vars.hidden

        start_life = episode_vars.start_life
        episode_end = episode_vars.episode_end
        dead = episode_vars.dead
        cur_time = episode_vars.cur_time
        stats = episode_vars.stats
            
        # set initial_h for the update
        initial_h = hidden        
            
        while not episode_end:
            cur_time += 1
                
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
                
            # append_sample(s, a, r, v)
            self.append_sample(state, action, reward, value)
                
            # if dead
            if start_life > info['ale.lives']:
                start_life = info['ale.lives']
                dead = True
                
            if cur_time >= self.t_max or episode_end:
                # Update the global network using samples during the episode.
                # Then, the vars of the global network is copied to the local network. 
                self.update_network(sess, initial_h, next_state, next_hidden, episode_end)
                sess.run(self.update_ops_holder)

                initial_h = next_hidden
                cur_time = 0
                # After updating, reset lists
                self.list_reset()
                    
            # change state to next_state
            state = next_state
            # change hidden to next_hidden
            hidden = next_hidden
            
        # end episode
        print("{} Episode done. avg_max_q: {:.4f}, total_reward: {} episode_step: {}".format(
            self.thread_var_scope, stats['avg_max_q']/stats['episode_step'], stats['total_reward'], int(stats['episode_step'])))
        self.update_summary_vars(sess, stats)
            
    def update_summary_vars(self, sess, stats):
        """Update summary."""
        stats['avg_max_q'] /= stats['episode_step']
        sess.run([self.summary_assign_ops[tag] for tag in self.summary_tags],
                 feed_dict={self.summary_placeholders[tag]:value for tag, value in stats.items()})
    
    def update_network(self, sess, first_h, last_s, last_h, episode_end):
        """Update network."""
        target = self.discount_rewards(sess, self.r_list, episode_end, last_s, last_h)
        assert len(target) == len(self.v_list)
        advantage = target - self.v_list
        
        sess.run(self.update_op, feed_dict={self.state:self.s_list,
                                            self.action:self.a_list,
                                            self.h_in:first_h,
                                            self.advantage:advantage,
                                            self.target:target})
        
    def linear(self, inputs, num_outputs, activation_fn=None, scope='linear'):
        "create linear layer"
        return model_helper.linear(
            inputs=inputs, 
            num_outputs=num_outputs, 
            w_init=self.w_init, 
            b_init=self.b_init, 
            activation_fn=activation_fn, 
            scope=scope)

    def conv2d(self, inputs, num_outputs, f_height, f_width, stride,
               activation_fn=tf.nn.relu, scope="conv2d"):    
        "create conv_2d layer"
        return model_helper.conv2d(
            inputs=inputs, 
            num_outputs=num_outputs, 
            f_height=f_height, 
            f_width=f_width, 
            w_init=self.w_init, 
            b_init=self.b_init, 
            stride=stride, 
            padding=self.padding,
            activation_fn=activation_fn,
            scope=scope)
       
    def create_rnn_cell(self):
        "create rnn cell"
        return model_helper.create_rnn_cell(
            num_layers=self.rnn_num_layers, 
            cell_type=self.cell_type, 
            num_units=self.num_units, 
            dropout=0.0, 
            mode=tf.estimator.ModeKeys.TRAIN, 
            residual_connect=self.residual_connect, 
            residual_fn=None, 
            num_gpus=self.num_gpus)
    
    def flatten(self, inputs):
        "flatten inputs"
        return model_helper.flatten(inputs)
