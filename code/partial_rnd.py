import numpy as np
from learner import DoubleQLearner
import tensorflow as tf
from welford import Welford

gamma_ext = .99
gamma_int = .9
eps0 = 0.3
episodes = 10000
runs = 20
max_grad_norm = 1.0
Q_int_coeff = 0.25
num_init_episodes = 1000
max_timesteps = 50

# changes I made to try to make this behave like neural.py
    # Q_int_coeff = 0, to take actions only wrt Q_ext
    # Q_loss = loss_ext, to not alter weights to better predict intrinsic reward
    # loss = Q_loss, to not alter weights to better predict fixed target net output

class RNDLearner(DoubleQLearner):
    def __init__(self, episodes, runs):
        super().__init__(episodes, runs)    # sets environment and handles visualization if turned on
        self.step_taker = self.q_learning   # the function that provides experience and does the learning
        self.window_area = self.env.window_size**2
        self.num_actions = len(self.env.action_space.actions)   # number of possible actions in the environment
        self.int_reward_stats = Welford()   # maintains running mean and variance of intrinsic reward
        self.obs_stats = Welford()  # maintains running mean and variance of observations (only for prediction and target networks)
        
        # placeholders
        self.inputs_ph = tf.placeholder(tf.float32, shape=(1,self.window_area))  # the observation at each time step
        self.aux_inputs_ph = tf.placeholder(tf.float32, shape=(1,self.window_area))  # inputs to prediction and fixed target networks
        self.targets_ext_ph = tf.placeholder(tf.float32, shape=(1,self.num_actions))    # r+gamma*Q(s',a'). external. q-learning
        self.targets_int_ph = tf.placeholder(tf.float32, shape=(1,self.num_actions))    # r+gamma*Q(s',a'). internal. q-learning

        # building the graph
        self.Q_ext, self.Q_int, self.Q_loss = self.build_graph() # the ouputs of Q network and its update operation
        self.aux_loss = self.build_aux_graphs()   # the internal reward and the prediciton net's update op
        self.update_op = self.update()
        self.init_op = tf.global_variables_initializer()    # global initialization done after the graph is defined
        
        # session
        self.saver = tf.train.Saver()   # for saving and restoring model weights
        self.sess = tf.Session()    # using the same session for the life of this RNDLearner object (each run). 
        self.sess.run(self.init_op) # initialize all model parameters
        self.saver.save(self.sess, "/tmp/model.ckpt") # save initial parameters


    # builds the computation graph for a Q network
    def build_graph(self):
        # separate Q-value heads for estimates of extrinsic and intrinsic returns
        Q_h = tf.layers.dense(self.inputs_ph, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="Q_h")
        Q_ext = tf.layers.dense(Q_h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="Q_ext")
        Q_int = tf.layers.dense(Q_h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="Q_int")

        loss_ext = tf.reduce_sum(tf.square(self.targets_ext_ph - Q_ext)) # error in prediction of external return
        loss_int = tf.reduce_sum(tf.square(self.targets_int_ph - Q_int)) # error in prediction of internal return
        Q_loss = loss_ext + loss_int
        return Q_ext, Q_int, Q_loss


    # defines the graph structure used for both the prediction net and the fixed target net
    def aux_graph(self, trainable):
        h1 = tf.layers.dense(self.aux_inputs_ph, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="h1", 
                trainable=trainable)
        output = tf.layers.dense(h1, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="output", 
                trainable=trainable)
        return output


    # returns operations for getting the prediction net loss (aka internal reward) and updating the prediction net parameters
    def build_aux_graphs(self):
        with tf.variable_scope("target_net"):
            target_net_out = self.aux_graph(trainable=False)
        with tf.variable_scope("predictor_net"):
            predictor_net_out = self.aux_graph(trainable=True)

        aux_loss = tf.reduce_sum(tf.square(predictor_net_out - target_net_out)) # loss for training predictor network. also intrinsic reward
        return aux_loss


    def update(self):
        loss = self.Q_loss + self.aux_loss
        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        update_op = optimizer.apply_gradients(zip(gradients, variables))
        return update_op


    # returns eps-greedy action with respect to Q_ext+Q_int
    def get_eps_action(self, obs, eps):
        Q_ext, Q_int = self.sess.run([self.Q_ext, self.Q_int], {self.inputs_ph: obs})  # outputs of Q network
        Q = Q_ext + Q_int_coeff*Q_int
        if self.env.action_space.np_random.uniform() < eps: # take random action with probability epsilon
            action = self.env.action_space.sample()
        else:
            max_actions = np.where(np.ravel(Q) == Q.max())[0]   # list of optimal action indices
            if len(max_actions) == 0:   # for debugging
                print(Q)
            action = self.env.action_space.np_random.choice(max_actions) # select from optimal actions randomly
        return action, Q_ext, Q_int


    # take step in environment and gather/update information
    def step(self, eps):
        obs = self.env.obs()
        action, Q_ext, Q_int = self.get_eps_action(obs, eps)    # take action wrt Q_ext+Q_int.
        next_obs, reward_ext, done, _ = self.env.step(action)  # get external reward by acting in the environment

        self.obs_stats.update(next_obs)  # update observation statistics
        whitened_next_obs = (next_obs - self.obs_stats.mean)/np.sqrt(self.obs_stats.var) # whitened obs for pred and target nets
        whitened_next_obs = np.clip(whitened_next_obs, -5, 5)

        reward_int = self.sess.run(self.aux_loss, {self.aux_inputs_ph: whitened_next_obs})  # get intrinsic reward
        self.int_reward_stats.update(reward_int)    # update running statistics for intrinsic reward
        reward_int = reward_int/np.sqrt(self.int_reward_stats.var)  # normalize intrinsic reward

        return obs, whitened_next_obs, action, reward_ext, reward_int, next_obs, Q_ext, Q_int, done


    def q_learning(self):
        self.saver.restore(self.sess, "/tmp/model.ckpt")  # restore the initial weights for each new run
        self.initialize_stats() # reset all statistics to zero and then initialize with random agent

        for episode in range(episodes):
            eps = eps0 - eps0*episode/episodes # decay epsilon
            done = False
            t = 0
            while not done: # for each episode
                t += 1

                # take step in environment and gather/update information
                obs, whitened_next_obs, action, reward_ext, reward_int, next_obs, Q_ext, Q_int, done = self.step(eps)
               
                #print(episode, reward_int)

                if t>max_timesteps:    # cap episode length at 20 timesteps
                    self.env.reset()
                    done = True
                
                # report data for analysis/plotting/visualization
                yield obs, self.env.action_space.actions[action], reward_ext, reward_int, next_obs, done

                # greedy next action wrt Q_ext+Q_int
                _, Q_ext_next, Q_int_next = self.get_eps_action(next_obs, 0)    
                
                # intrinsic reward is non-episodic
                target_value_int = reward_int + gamma_int*np.max(Q_int_next)    
       
                # extrinsic reward is episodic
                target_value_ext = reward_ext
                if not done:
                    target_value_ext += gamma_ext*np.max(Q_ext_next)    

                target_Q_ext = Q_ext    # only chosen action can have nonzero error
                target_Q_ext[0,action] = target_value_ext   # the first index is into the zeroth (and only) batch dimension

                target_Q_int = Q_int
                target_Q_int[0,action] = target_value_int

                # update all parameters to minimize combined loss
                self.sess.run(self.update_op, {self.inputs_ph: obs, self.targets_ext_ph: target_Q_ext, 
                    self.targets_int_ph: target_Q_int, self.aux_inputs_ph: whitened_next_obs})


    # initialize intrinsic reward and observation statistics with experience from a random agent
    def initialize_stats(self):
        # reset statistics to zero
        self.int_reward_stats.reset()
        self.obs_stats.reset()
       
        # initialize observation stats first
        obs_list = []
        for episode in range(num_init_episodes):
            done = False
            while not done:
                action = self.env.action_space.sample() # take an action to give us another observation
                next_obs, _, done, _ = self.env.step(action)  
                self.obs_stats.update(next_obs)  # update obs stats
                obs_list.append(next_obs)    # save to list for use with intrinsic reward stats

        # create whitened observations for input to the aux networks
        for observation in obs_list:
            whitened_next_obs = (observation - self.obs_stats.mean)/np.sqrt(self.obs_stats.var) # whitened obs for pred and target nets
            whitened_next_obs = np.clip(whitened_next_obs, -5, 5)
            reward_int = self.sess.run(self.aux_loss, {self.aux_inputs_ph: whitened_next_obs})  # get intrinsic reward
            self.int_reward_stats.update(reward_int)    # update int_reward stats


learner = RNDLearner(episodes, runs)
learner.main()
