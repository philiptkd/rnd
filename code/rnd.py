import numpy as np
from learner import DoubleQLearner
import tensorflow as tf

alpha = 0.1
gamma = .99
eps0 = 0.3
episodes = 5000 
runs = 20
c = 2
lr = .1
visualize = False

class RNDLearner(DoubleQLearner):
    def __init__(self, episodes, runs):
        super().__init__(episodes, runs)
        self.step_taker = self.q_learning
        self.num_actions = len(self.env.action_space.actions)
        self.num_states = self.env.width*self.env.height
        self.inputs = tf.placeholder(tf.float32, shape=(1,self.num_states))
        self.targets_ext = tf.placeholder(tf.float32, shape=(1,self.num_actions))
        self.targets_int = tf.placeholder(tf.float32, shape=(1,self.num_actions))
        self.Q_ext_out_op, self.Q_int_out_op, self.Q_update_op = self.build_graph()
        self.int_reward_op, self.pred_update_op = self.build_aux_graphs()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.saver.save(self.sess, "/tmp/model.ckpt") # save initial parameters

    # builds the computation graph for a Q network
    def build_graph(self):
        # separate Q-value heads for estimates of extrinsic and intrinsic returns
        Q_h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="Qh")
        Q_ext = tf.layers.dense(Q_h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="Q_ext")
        Q_int = tf.layers.dense(Q_h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="Q_int")

        loss_ext = tf.reduce_sum(tf.square(self.targets_ext - Q_ext)) # error in prediction of external return
        loss_int = tf.reduce_sum(tf.square(self.targets_int - Q_int)) # error in prediction of internal return
        loss = loss_ext + loss_int

        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        update = optimizer.apply_gradients(zip(gradients, variables))
        return Q_ext, Q_int, update

    def aux_graph(self, trainable):
        h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="h", 
                trainable=trainable)
        output = tf.layers.dense(h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="output", 
                trainable=trainable)
        return output

    def build_aux_graphs(self):
        with tf.variable_scope("target_net"):
            target_net_out = self.aux_graph(trainable=False)
        with tf.variable_scope("predictor_net"):
            predictor_net_out = self.aux_graph(trainable=True)

        loss_pred = tf.reduce_sum(tf.square(predictor_net_out - target_net_out)) # loss for training predictor network. also intrinsic reward
        # add gradient clipping?
        update_pred = tf.train.AdamOptimizer().minimize(loss)
        return loss_pred, update_pred

    #TODO:  self.targets_int is the MSE between prediction and target net outputs
        #       i.e. do a q-learning update for the Q_int value head
        #   normalize returns and observations

    # returns eps-greedy action with respect to Q
    def get_eps_action(self, one_hot_state, eps):
        Q = self.sess.run(self.Q_out_op, {self.inputs: one_hot_state})
        if self.env.action_space.np_random.uniform() < eps:
            action = self.env.action_space.sample()
        else:
            max_actions = np.where(np.ravel(Q) == Q.max())[0]
            if len(max_actions) == 0:
                print(Q)
            action = self.env.action_space.np_random.choice(max_actions) # to select argmax randomly
        return action, Q

    def q_learning(self):
        self.saver.restore(self.sess, "/tmp/model.ckpt")  # restore the initial weights for each new run
        for episode in range(episodes):
            eps = eps0 - eps0*episode/episodes # decay epsilon
            done = False
            t = 0
            while not done:
                t += 1
                state = self.env.state
                action, Q = self.get_eps_action(self.one_hot(state), eps)
                _, reward, done, _ = self.env.step(action)
                next_state = self.env.state
                yield self.env.action_space.actions[action], reward, done
                
                # q-learning update
                target_value = reward
                if not done:
                    _, Q_next = self.get_eps_action(self.one_hot(next_state), 0)    # greedy action
                    target_value += gamma*np.max(Q_next)
                target_Q = Q    # only chosen action can have nonzero error
                target_Q[0,action] = target_value
                self.sess.run(self.Q_update_op, {self.inputs: self.one_hot(state), self.targets: target_Q})

                if t>20:
                    self.env.reset()
                    break

    # returns one-hot representation of the given state
    def one_hot(self, state):
        one_hot_state = np.zeros(self.num_states)
        one_hot_state[state[0]*self.env.width + state[1]] = 1
        return np.array([one_hot_state])    # add batch dimension of length 1

learner = NeuralLearner(episodes, runs)
learner.main()
