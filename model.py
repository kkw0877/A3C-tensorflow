import gym
import tensorflow as tf

import random
import numpy as np
from collections import namedtuple

import model_helper
from utils import misc_utils



class TrainOutput(namedtuple(
    "TrainOutput", ("global_step", "train_summary"))):
    pass

class InferOutput(namedtuple(
    "InferOutput", ("infer_summary"))):
    pass

class EpisodeVars(namedtuple(
    "EpisodeVars", ("state", "hidden", 
                    "start_life", "episode_end", 
                    "dead", "cur_time", "stats"))):
    pass




class PolicyGradient(object):
    """Policy gradient base class."""
    def __init__(self, flags, mode):
        # Set hparams
        self.set_hparams_init(flags, mode)
        
        # Build graph
        res = self.build_graph(flags) 
        self.set_train_or_infer(res, flags)
        
        # Saver 
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=flags.num_keep_ckpts)
    
    def set_hparams_init(self, flags, mode):
        # Select mode for TRAIN | PREDICT(infer)
        self.mode = mode
        
        # Set environment 
        self.env = gym.make(flags.env)
        # [batch_size, image_height, image_width, num_channels]
        self.state = tf.placeholder(tf.float32, 
            [None, flags.img_height, flags.img_width, 1]) 
        # possible_action: [stop, left, right]
        self.action_size = 3 
        
        # Set image height and width
        self.img_height = flags.img_height
        self.img_width = flags.img_width
        
        # Global step
        self.global_step = tf.Variable(0, trainable=False)
        
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
        
    def build_graph(self, flags):
        """Build policy gradient graph"""
        with tf.variable_scope(flags.model_name):
            # TODO. will fix tf.device for multiple gpu
            with tf.device(model_helper.get_device_str(0, self.num_gpus)):
                c1 = self.conv2d(self.state, self.cv_num_outputs, 
                                 self.f_height, self.f_width, 
                                 self.stride, scope="conv2d_1")
                c2 = self.conv2d(c1, self.cv_num_outputs*2, 
                                 self.f_height//2, self.f_width//2, 
                                 self.stride//2, scope="conv2d_2")
                fc = self.linear(self.flatten(c2), 
                        self.num_units, activation_fn=tf.nn.relu, scope='flat')
                
                # modify the shape of the fc before rnn
                # [1, None, self.flat_outputs]
                rnn_input = tf.reshape(fc, [1, -1, self.num_units]) 
                step_size = tf.shape(rnn_input)[1:2] 

                rnn_cell = self.create_rnn_cell()
                self.h_in = rnn_cell.zero_state(1, tf.float32)

                rnn_output, self.h_out = tf.nn.dynamic_rnn(
                    rnn_cell, rnn_input, initial_state=self.h_in, 
                    sequence_length=step_size)
                rnn_output = tf.reshape(rnn_output, [-1, self.num_units])

                # policy 
                self.policy = self.linear(
                    rnn_output, self.action_size, 
                    activation_fn=tf.nn.softmax, scope='policy')

                # compute loss
                if self.mode != tf.estimator.ModeKeys.PREDICT:
                    loss = self.compute_loss()
                else:
                    loss = tf.constant(0.0)
                
                return loss
            
    def compute_loss(self):
        """Compute Optimization loss"""
        policy = self.policy
        self.action = tf.placeholder(tf.int32, [None])
        self.advantage = tf.placeholder(tf.float32, [None])
        
        actions_onehot = tf.one_hot(self.action, self.action_size)
        action_probability = tf.reduce_sum(policy * actions_onehot, axis=1)

        # actor loss
        actor_loss = -tf.reduce_sum(
            tf.log(action_probability) * self.advantage)

        # entropy loss for exploration
        entropy_loss = tf.reduce_sum(policy * tf.log(policy+1e-10), axis=1)
        entropy_loss = tf.reduce_sum(entropy_loss)
        
        total_loss = actor_loss + entropy_loss
        return total_loss
    
    def set_train_or_infer(self, res, flags):
        """Set up train and infer"""
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            # set train_loss, learning rate, discount factor for train
            train_loss = res
            self.learning_rate = flags.learning_rate
            self.discount_factor = flags.discount_factor
            
            params = tf.trainable_variables()
        
            if flags.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self.learning_rate)
            elif flags.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
            else:
                raise ValueError("Unknown optimizer %s" % flags.optimizer)

            gradients = tf.gradients(train_loss, params)
            clipped_gradients, global_norm = tf.clip_by_global_norm(
                gradients, flags.max_gradient_norm)
            self.update_op = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)    
            
        self.summary = self.get_summary()
        
    def get_summary(self):
        """Get summary """
        # To know the performance of the model during episode,
        # set variables except for loss
        self.summary_tags = ['episode_step', 
                             'avg_max_q', 
                             'total_reward']
        
        self.summary_vars = {}
        self.summary_placeholders = {}
        self.summary_assign_ops = {}
        
        for tag in self.summary_tags:
            self.summary_vars[tag] = tf.Variable(0., trainable=False)
            self.summary_placeholders[tag] = tf.placeholder(tf.float32)
            self.summary_assign_ops[tag] = self.summary_vars[tag].assign(
                self.summary_placeholders[tag])
            tf.summary.scalar('%s' % tag, self.summary_vars[tag])
            
        summary = tf.summary.merge_all()
        return summary
    
    def get_action(self, sess, state, hidden):
        """Get action through the network"""
        policy, next_hidden = sess.run([self.policy, self.h_out], 
            feed_dict={self.state:[state], self.h_in:hidden})
        policy = policy[0] 
        action = np.random.choice(self.action_size, p=policy)
        return action, policy, next_hidden
        
    def discount_rewards(self, rewards):
        """Get discount rewards to update the network."""
        value = 0
        discount_rewards = np.ones_like(rewards)
        
        for i in reversed(range(len(rewards))):
            value = rewards[i] + self.discount_factor * value
            discount_rewards[i] = value
            
        return discount_rewards

    def append_sample(self, s, a, r):
        self.s_list.append(s)
        self.a_list.append(a)
        self.r_list.append(r) 
        
    def list_reset(self):
        """reset s_list, a_list, r_list for train mode."""
        self.s_list = [] # state
        self.a_list = [] # action
        self.r_list = [] # reward
        
    def train(self, sess):
        """Execute train graph."""
        assert self.mode == tf.estimator.ModeKeys.TRAIN
        
        # reset the lists before the episode 
        self.list_reset()
        
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
            # get action, policy and next_hidden through the network
            action, policy, next_hidden = self.get_action(sess, state, hidden)
            
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
            
            # update stats about duration, avg_max_q, total_reward
            self.update_stats(stats, reward, policy)
            
            # append sample(state, action, reward)
            self.append_sample(state, action, reward)
            
            # if dead
            if start_life > info['ale.lives']:
                start_life = info['ale.lives']
                dead = True
      
            # change state to next_state
            state = misc_utils.preprocess_image(next_state, self.img_height, self.img_width)
            # change hidden to next_hidden
            hidden = next_hidden
        
        print("Episode done. avg_max_q: {:.4f}, total_reward: {}".format(
            stats['avg_max_q']/stats['episode_step'], stats['total_reward']))

        # When the episode ends, update network and summary_vars 
        # then, return the train result
        self.update_network(sess, initial_h)
        self.update_summary_vars(sess, stats)
        train_output = TrainOutput(global_step=self.global_step, train_summary=self.summary)
        return sess.run(train_output)

    def before_episode(self, sess):
        """ initialize vars related to the Atari environment before start."""
        state = self.env.reset()
        for _ in range(random.randint(1, 30)):
            state, _, _, _ = self.env.step(1)
        state = misc_utils.preprocess_image(state, self.img_height, self.img_width)
        hidden = sess.run(self.h_in)
        
        # start_life, episode_end, dead, cur_time
        start_life = 5
        episode_end = False
        dead = False
        cur_time = 0 # only use for A3C
        
        # the model performance
        stats = {tag: 0.0 for tag in self.summary_tags}
        
        # set episode_vars related to the episode
        episode_vars = EpisodeVars(state=state,
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
        
    def update_summary_vars(self, sess, stats):
        """Update summary."""
        stats['avg_max_q'] /= stats['episode_step']
        sess.run([self.summary_assign_ops[tag] for tag in self.summary_tags],
                 feed_dict={self.summary_placeholders[tag]:value for tag, value in stats.items()})
        
    def update_network(self, sess, initial_h):
        """Update network."""
        discount_rewards = self.discount_rewards(self.r_list)
        sess.run(self.update_op, feed_dict={self.state:self.s_list,
                                            self.h_in:initial_h,
                                            self.action:self.a_list,
                                            self.advantage:discount_rewards})
        
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
            # get action, policy and next_hidden through the network
            action, policy, next_hidden = self.get_action(sess, state, hidden)

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

            # update stats about duration, avg_max_q, total_reward
            self.update_stats(stats, reward, policy)
            
            # if dead
            if start_life > info['ale.lives']:
                start_life = info['ale.lives']
                dead = True
            
            # change state to next_state
            state = misc_utils.preprocess_image(next_state, self.img_height, self.img_width)
            # change hidden to next_hidden
            hidden = next_hidden
            
        # end episode
        print("Infer_mode done. avg_max_q: {:.4f}, total_reward: {}".format(
            stats['avg_max_q']/stats['episode_step'], stats['total_reward']))
        self.update_summary_vars(sess, stats)
        infer_output = InferOutput(infer_summary=self.summary)

        return sess.run(infer_output)
    
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
            dropout=self.dropout, 
            mode=self.mode, 
            residual_connect=self.residual_connect, 
            residual_fn=None, 
            num_gpus=self.num_gpus)
    
    def flatten(self, inputs):
        "flatten inputs"
        return model_helper.flatten(inputs)
