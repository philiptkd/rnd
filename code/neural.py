import numpy as np
from learner import DoubleQLearner
import tensorflow as tf

alpha = 0.1
gamma = 1
eps = 0.1
episodes = 2000 
runs = 1
c = 2
lr = .1
visualize = False

class NeuralLearner(DoubleQLearner):
    def __init__(self, episodes, runs):
        super().__init__(episodes, runs)
        self.step_taker = self.q_learning
        self.reset_graph()
        
    # clears the graph and recreates it. for restarting learning in a new run.
    def reset_graph(self):
        tf.reset_default_graph()

        # placeholders
        self.inputs = tf.placeholder(tf.float32, shape=(1,self.env.width*self.env.height))
        self.targets = tf.placeholder(tf.float32, shape=(1,len(self.env.actions)))
       
        # operations
        self.Q_out_op, self.Q_update_op = self.build_graph()
        
        # initialize variables
        self.init_op = tf.global_variables_initializer()


    # builds the computation graph for a Q network
    def build_graph(self):
        h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, name="h")
        outputs = tf.layers.dense(h, len(self.env.actions), activation=None, name="outputs")
        loss = tf.reduce_sum(tf.square(self.targets - outputs))
        update = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        return outputs, update

    # returns eps-greedy action with respect to Q
    def get_eps_action(self, one_hot_state, eps=eps):
        Q = self.sess.run(self.Q_out_op, {self.inputs: one_hot_state})
        if self.env.np_random.uniform() < eps:
            action = self.env.np_random.randint(0, len(self.env.actions))
        else:
            action = self.env.np_random.choice(np.flatnonzero(Q == Q.max())) # to select argmax randomly
        return action, Q

    def q_learning(self):
        self.reset_graph()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        for episode in range(episodes):
            done = False
            while not done:
                state = self.env.state
                action, Q = self.get_eps_action(self.one_hot(state))
                reward, done = self.env.step(self.env.actions[action])
                next_state = self.env.state
                yield self.env.actions[action], reward, done
                
                # q-learning update
                target_value = reward
                if not done:
                    _, Q_next = self.get_eps_action(self.one_hot(next_state), eps=0)    # greedy action
                    target_value += gamma*np.max(Q_next)
                target_Q = Q    # only chosen action can have nonzero error
                target_Q[0,action] = target_value
                self.sess.run(self.Q_update_op, {self.inputs: self.one_hot(state), self.targets: target_Q})

        self.sess.close()

    # returns one-hot representation of the given state
    def one_hot(self, state):
        one_hot_state = np.zeros(self.env.width*self.env.height)
        one_hot_state[state[0]*self.env.width + state[1]] = 1
        return np.array([one_hot_state])    # add batch dimension of length 1

learner = NeuralLearner(episodes, runs)
learner.main()
