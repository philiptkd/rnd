import numpy as np
from learner import DoubleQLearner

alpha = 0.1
gamma = 1
eps = 0.3
episodes = 10000
runs = 100
c = 2
visualize = False

class TabularLearner(DoubleQLearner):
    def __init__(self):
        super().__init__()
        self.step_taker = self.double_q

    def double_q(self, single_q=False):
        self.Q1 = np.zeros((self.env.width*self.env.height, len(self.env.actions)))
        self.Q2 = np.zeros((self.env.width*self.env.height, len(self.env.actions)))
        
        counts = np.ones((self.env.width*self.env.height, len(self.env.actions)))
        num_steps = 1

        for episode in range(episodes):
            done = False

            while not done:
                state = self.env.state
                action = get_ucb_action(self.env, state, self.Q1+self.Q2, counts, num_steps)
                reward, done = self.env.step(self.env.actions[action])
                next_state = self.env.state

                yield self.env.actions[action], reward, done
                
                step = (state,action,reward,next_state,done)
                if single_q:
                    double_q_update(self.Q1, self.Q1, self.env, step, counts, num_steps)
                elif env.np_random.uniform() < 0.5:
                    double_q_update(self.Q1, self.Q2, self.env, step, counts, num_steps)
                else:
                    double_q_update(self.Q2, self.Q1, self.env, step, counts, num_steps)

                num_steps += 1
                counts[s2idx(self.env, state), action] += 1

# converts environment state to Q index
def s2idx(env, state):
    return state[0]*env.width + state[1]

# gets exploration bonus
def get_bonus(counts, num_steps):
    return c*np.sqrt(np.log(num_steps)/counts)

# update self.Q1 towards self.Q2 estimate of value of Q1's max action
def double_q_update(Q1, Q2, env, step, counts, num_steps):
    state, action, reward, next_state, done = step
    target = reward
    if not done:
        next_action = get_ucb_action(env, next_state, Q1, counts, num_steps)
        target += gamma*Q2[s2idx(env, next_state), next_action]
    Q1[s2idx(env, state), action] += alpha*(target - Q1[s2idx(env, state), action])

# returns eps-greedy action
def get_eps_action(env, state, Q, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.randint(0, len(env.actions))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[s2idx(env, state)] == Q[s2idx(env, state)].max())) # to select argmax randomly
    return action

# returns UCB optimistic action.
def get_ucb_action(env, state, Q, counts, num_steps):
    bonus = get_bonus(counts, num_steps)
    ucb = Q + bonus
    action = env.np_random.choice(np.flatnonzero(ucb[s2idx(env, state)] == ucb[s2idx(env, state)].max())) # to select argmax randomly
    return action

learner = TabularLearner()
learner.main()
