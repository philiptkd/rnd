import numpy as np
from learner import DoubleQLearner
import tensorflow as tf
from welford import Welford

alpha = 0.1
gamma_ext = .999
gamma_int = .99
eps0 = 0.3
episodes = 5000 
runs = 20
c = 2
lr = .1

class RNDLearner(DoubleQLearner):
    def __init__(self, episodes, runs):
        super().__init__(episodes, runs)    # sets environment and handles visualization if turned on
        self.step_taker = self.q_learning   # the function that provides experience and does the learning
        self.num_actions = len(self.env.action_space.actions)   # number of possible actions in the environment
        self.num_states = self.env.width*self.env.height        # number of states in the environment
        self.int_reward_stats = Welford()   # maintains running mean and variance of intrinsic reward
        self.obs_stats = Welford()  # maintains running mean and variance of observations (only for prediction and target networks)
        
        # placeholders
        self.inputs_ph = tf.placeholder(tf.float32, shape=(1,self.num_states))  # the observation at each time step
        self.aux_inputs_ph = tf.placeholder(tf.float32, shape=(1,self.num_states))  # inputs to prediction and fixed target networks
        self.targets_ext_ph = tf.placeholder(tf.float32, shape=(1,self.num_actions))    # r+gamma*Q(s',a'). external. q-learning
        self.targets_int_ph = tf.placeholder(tf.float32, shape=(1,self.num_actions))    # r+gamma*Q(s',a'). internal. q-learning

        # operations
        self.Q_ext_out_op, self.Q_int_out_op, self.Q_update_op = self.build_graph() # the ouputs of Q network and its update operation
        self.int_reward_op, self.pred_update_op = self.build_aux_graphs()   # the internal reward and the prediciton net's update op
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
        loss = loss_ext + loss_int

        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)   # gradient clipping because of the scale of external rewards (I think)
        Q_update = optimizer.apply_gradients(zip(gradients, variables))
        return Q_ext, Q_int, Q_update


    # defines the graph structure used for both the prediction net and the fixed target net
    def aux_graph(self, trainable):
        h = tf.layers.dense(self.aux_inputs_ph, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="h", 
                trainable=trainable)
        output = tf.layers.dense(h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="output", 
                trainable=trainable)
        return output


    # returns operations for getting the prediction net loss (aka internal reward) and updating the prediction net parameters
    def build_aux_graphs(self):
        with tf.variable_scope("target_net"):
            target_net_out = self.aux_graph(trainable=False)
        with tf.variable_scope("predictor_net"):
            predictor_net_out = self.aux_graph(trainable=True)

        loss_pred = tf.reduce_sum(tf.square(predictor_net_out - target_net_out)) # loss for training predictor network. also intrinsic reward
        # TODO: add gradient clipping?
        update_pred = tf.train.AdamOptimizer().minimize(loss_pred)
        return loss_pred, update_pred


    # returns eps-greedy action with respect to Q_ext+Q_int
    def get_eps_action(self, one_hot_state, eps):
        Q_ext, Q_int = self.sess.run([self.Q_ext_out_op, self.Q_int_out_op], {self.inputs_ph: one_hot_state})  # outputs of Q network
        Q = Q_ext + Q_int
        if self.env.action_space.np_random.uniform() < eps: # take random action with probability epsilon
            action = self.env.action_space.sample()
        else:
            max_actions = np.where(np.ravel(Q) == Q.max())[0]   # list of optimal action indices
            if len(max_actions) == 0:   # for debugging
                print(Q)
            action = self.env.action_space.np_random.choice(max_actions) # select from optimal actions randomly
        return action, Q_ext, Q_int


    # take step in environment and gather/update information
    def step(self):
        state = self.env.state
        one_hot = self.one_hot(state)   # one-hot representation of current state
        self.obs_stats.update(one_hot)  # update observation statistics
        whitened_state = (one_hot - self.obs_stats.mean)/np.sqrt(self.obs_stats.var) # whitened observation for pred and target nets

        action, Q_ext, Q_int = self.get_eps_action(one_hot, eps)    # take action wrt Q_ext+Q_int
        _, reward_ext, done, _ = self.env.step(action)  # get external reward by acting in the environment

        reward_int = self.sess.run(self.int_reward_op, {self.aux_inputs_ph: whitened_state})  # get intrinsic reward
        self.int_reward_stats.update(reward_int)    # update running statistics for intrinsic reward
        reward_int = reward_int/np.sqrt(self.int_reward_stats.var)  # normalize intrinsic reward
        
        next_state = self.env.state
        yield self.env.action_space.actions[action], reward_ext, reward_int, done   # report data for analysis/plotting/visualization
        
        return state, whitened_state, action, reward_ext, reward_int, next_state, Q_ext, Q_int


    def q_learning(self):
        self.saver.restore(self.sess, "/tmp/model.ckpt")  # restore the initial weights for each new run
        self.int_reward_stats.reset()   # get rid of old statistics for each new run
        self.obs_stats.reset()

        for episode in range(episodes):
            eps = eps0 - eps0*episode/episodes # decay epsilon
            done = False
            t = 0
            while not done: # for each episode
                t += 1

                # take step in environment and gather/update information
                state, whitened_state, action, reward_ext, reward_int, next_state, Q_ext, Q_int = self.step()

                # q-learning update for both Q-value heads
                target_value_ext = reward_ext
                target_value_int = reward_int
                if not done:
                    _, Q_ext_next, Q_int_next = self.get_eps_action(self.one_hot(next_state), 0)    # greedy action wrt Q_ext+Q_int
                    target_value_ext += gamma_ext*np.max(Q_ext_next)
                    target_value_int += gamma_int*np.max(Q_int_next)

                target_Q_ext = Q_ext    # only chosen action can have nonzero error
                target_Q_ext[0,action] = target_value_ext   # the first index is into the zeroth (and only) batch dimension

                target_Q_int = Q_int
                target_Q_int[0,action] = target_value_int

                # update Q network through both Q-value heads
                self.sess.run(self.Q_update_op, {self.inputs_ph: one_hot, self.targets_ext_ph: target_Q_ext, 
                    self.targets_int_ph: target_Q_int})

                # update predictor network
                self.sess.run(self.pred_update_op, {self.aux_inputs_ph: whitened_state})

                if t>20:    # cap episode length at 20 timesteps
                    self.env.reset()
                    break

    # returns one-hot representation of the given state
    def one_hot(self, state):
        one_hot_state = np.zeros(self.num_states)
        one_hot_state[state[0]*self.env.width + state[1]] = 1
        return np.array([one_hot_state])    # add batch dimension of length 1

learner = NeuralLearner(episodes, runs)
learner.main()
