# a basic q-learning agent with full observability

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

class NeuralLearner(DoubleQLearner):
    def __init__(self, episodes, runs):
        super().__init__(episodes, runs)
        self.step_taker = self.q_learning
        self.num_actions = len(self.env.action_space.actions)
        self.num_states = self.env.width*self.env.height
        self.inputs = tf.placeholder(tf.float32, shape=(1,self.num_states))
        self.targets = tf.placeholder(tf.float32, shape=(1,self.num_actions))
        self.Q_out_op, self.Q_update_op = self.build_graph()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.saver.save(self.sess, "/tmp/model.ckpt") # save initial parameters

    # builds the computation graph for a Q network
    def build_graph(self):
        h = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal, name="h")
        outputs = tf.layers.dense(h, self.num_actions, activation=None, kernel_initializer=tf.initializers.random_normal, name="outputs")
        loss = tf.reduce_sum(tf.square(self.targets - outputs))
        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        update = optimizer.apply_gradients(zip(gradients, variables))
        return outputs, update

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
                yield state, self.env.action_space.actions[action], reward, 0, next_state, done # 0 is intrinsic reward
                
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
