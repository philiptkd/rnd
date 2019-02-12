import numpy as np
from learner import DoubleQLearner
import tensorflow as tf
from welford import Welford

gamma_ext = .99
eps0 = 0.3
episodes = 10000
runs = 20
max_grad_norm = 1.0
max_timesteps = 20

class NeuralLearner(DoubleQLearner):
    def __init__(self, episodes, runs, trail=False):
        super().__init__(episodes, runs, trail)    # sets environment and handles visualization if turned on
        self.step_taker = self.q_learning   # the function that provides experience and does the learning
        self.window_area = self.env.window_size**2
        self.num_actions = len(self.env.action_space.actions)   # number of possible actions in the environment
        
        # placeholders
        self.inputs_ph = tf.placeholder(tf.float32, shape=(1,self.window_area))  # the observation at each time step
        self.targets_ext_ph = tf.placeholder(tf.float32, shape=(1,self.num_actions))    # r+gamma*Q(s',a'). external. q-learning

        # building the graph
        self.Q_ext, self.Q_loss = self.build_graph() # the ouputs of Q network and its update operation
        self.update_op = self.update()
        self.init_op = tf.global_variables_initializer()    # global initialization done after the graph is defined
        
        # session
        self.saver = tf.train.Saver()   # for saving and restoring model weights
        self.sess = tf.Session()    # using the same session for the life of this RNDLearner object (each run). 
        self.sess.run(self.init_op) # initialize all model parameters
        self.saver.save(self.sess, "/tmp/model.ckpt") # save initial parameters


    # builds the computation graph for a Q network
    def build_graph(self):
        Q_h = tf.layers.dense(self.inputs_ph, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="Q_h")
        Q_ext = tf.layers.dense(Q_h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="Q_ext")
        loss_ext = tf.reduce_sum(tf.square(self.targets_ext_ph - Q_ext)) # error in prediction of external return
        Q_loss = loss_ext
        return Q_ext, Q_loss


    def update(self):
        loss = self.Q_loss
        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        update_op = optimizer.apply_gradients(zip(gradients, variables))
        return update_op


    # returns eps-greedy action with respect to Q_ext+Q_int
    def get_eps_action(self, obs, eps):
        Q_ext = self.sess.run(self.Q_ext, {self.inputs_ph: obs})  # outputs of Q network
        Q = Q_ext
        if self.env.action_space.np_random.uniform() < eps: # take random action with probability epsilon
            action = self.env.action_space.sample()
        else:
            max_actions = np.where(np.ravel(Q) == Q.max())[0]   # list of optimal action indices
            if len(max_actions) == 0:   # for debugging
                print(Q)
            action = self.env.action_space.np_random.choice(max_actions) # select from optimal actions randomly
        return action, Q_ext


    # take step in environment and gather/update information
    def step(self, eps):
        obs = self.env.obs()
        action, Q_ext = self.get_eps_action(obs, eps)    # take action wrt Q_ext+Q_int.
        next_obs, reward_ext, done, _ = self.env.step(action)  # get external reward by acting in the environment
        return obs, action, reward_ext, next_obs, Q_ext, done


    def q_learning(self):
        self.saver.restore(self.sess, "/tmp/model.ckpt")  # restore the initial weights for each new run

        for episode in range(episodes):
            eps = eps0 - eps0*episode/episodes # decay epsilon
            done = False
            t = 0
            while not done: # for each episode
                t += 1

                # take step in environment and gather/update information
                obs, action, reward_ext, next_obs, Q_ext, done = self.step(eps)
               
                #print(episode, reward_int)

                if t>max_timesteps:    # cap episode length at 20 timesteps
                    self.env.reset()
                    done = True
                
                # report data for analysis/plotting/visualization
                yield obs, self.env.action_space.actions[action], reward_ext, 0, next_obs, done

                # extrinsic reward is episodic
                target_value_ext = reward_ext
                if not done:
                    _, Q_ext_next = self.get_eps_action(next_obs, 0)    
                    target_value_ext += gamma_ext*np.max(Q_ext_next)    

                target_Q_ext = Q_ext    # only chosen action can have nonzero error
                target_Q_ext[0,action] = target_value_ext   # the first index is into the zeroth (and only) batch dimension

                # update all parameters to minimize combined loss
                self.sess.run(self.update_op, {self.inputs_ph: obs, self.targets_ext_ph: target_Q_ext})


learner = NeuralLearner(episodes, runs, trail=True)
learner.main()
