import numpy as np
from learner import DoubleQLearner
import tensorflow

alpha = 0.1
gamma = 1
eps = 0.3
episodes = 10000
runs = 100
c = 2
visualize = False

class NeuralLearner(DoubleQLearner):
    def __init__(self):
        super().__init__()
        self.step_taker = self.double_q

    def build_graph(self):
        inputs = tf.placeholder(tf.int32, shape=(self.env.width*self.env.height,))
        targets = tf.placeholder(tf.float32, shape=(len(self.env.actions),))
        h = tf.layers.dense(inputs, 64, activation=tf.nn.relu, name="h")
        dropout = tf.layers.dropout(h1, .2, name="dropout")
        outputs = tf.layers.dense(dropout, len(self.env.actions), activation=None, name="outputs")
        loss = tf.reduce_sum(tf.square(targets - outputs))
        update = tf.train.Adam().minimize(loss)
        return inputs, targets, outputs, update

    def get_eps_action(self, one_hot_state, Q1_out_op, Q2_out_op, eps=eps):
        if env.np_random.uniform() < eps:
            action = env.np_random.randint(0, len(env.actions))
        else:
            Q1 = self.sess.run([Q1_out_op], {self.inputs: one_hot_state})
            Q2 = self.sess.run([Q2_out_op], {self.inputs: one_hot_state})
            Q = Q1 + Q2
            action = env.np_random.choice(np.flatnonzero(Q == Q.max())) # to select argmax randomly
        return action, Q1, Q2

    def get_greedy_action(self, one_hot_state, Q_out_op):
        Q = self.sess.run([Q_out_op], {self.inputs: one_hot_state})
        action = env.np_random.choice(np.flatnonzero(Q == Q.max())) # to select argmax randomly
        return action, Q 

    def double_q(self, single_q=False):
        # defining the computational graph
        with tf.variable_scope("Q1"):
            self.inputs, self.targets, Q1_out_op, Q1_update_op = self.build_graph()
        with tf.variable_scope("Q2"):
            _, _, Q2_out_op, Q2_update_op = self.build_graph()
        init_op = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init_op)
        for episode in range(episodes):
            done = False
            while not done:
                # action is argmax of Q1+Q2 from forward pass through both networks                
                state = self.env.state
                action, Q1, Q2 = self.get_eps_action(self.one_hot(state), Q1_out_op, Q2_out_op)

                reward, done = self.env.step(self.env.actions[action])
                next_state = self.env.state

                yield self.env.actions[action], reward, done
                
                step = (state,action,reward,next_state,done)
                if single_q:
                    double_q_update(Q1, Q1_update_op, Q1_out_op, Q1_out_op, step)
                elif env.np_random.uniform() < 0.5:
                    double_q_update(Q1, Q1_update_op, Q1_out_op, Q2_out_op, step)
                else:
                    double_q_update(Q2, Q2_update_op, Q2_out_op, Q1_out_op, step)

    # update Q1 towards Q2 estimate of value of Q1's max action
    def double_q_update(self, Q1, Q1_update_op, Q1_out_op, Q2_out_op, step):
        state, action, reward, next_state, done = step
        action_target = reward
        if not done:
            next_action,_ = self.get_greedy_action(self.one_hot(next_state), Q1_out_op) # Q1's best next action
            _,Q2 = self.get_greedy_action(self.one_hot(next_state), Q2_out_op)
            action_target += gamma*Q2[next_action]  # Q1's best action evaluated by Q2
        q_target = Q1
        q_target[action] = action_target
        sess.run([Q1_update_op], {inputs: self.one_hot(state), targets: q_target})
        
    def one_hot(self, state):
        one_hot_state = np.zeros(self.env.width*self.env.height)
        one_hot_state[state[0]*self.env.width + state[1]] = 1
        return one_hot_state

