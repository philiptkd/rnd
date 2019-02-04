import numpy as np
import matplotlib.pyplot as plt
from clamity_env import ClamEnv

alpha = 0.1
gamma = 1
eps = 0.3
episodes = 10000
runs = 100
c = 2
visualize = False

class Learner():
    def __init__(self):
        self.env = ClamEnv()
        self.step_taker = self.double_q
        if visualize:
            from display import Disp
            self.disp = Disp(self.env)


    def main(self):
        avg_ep_returns = np.zeros(episodes) # to plot
        for run in range(runs):
            print(run)
            episode_returns = np.zeros(episodes)

            step = self.step_taker(self.env) # step generator. learning also happens
            G = 0
            episode = 0
            while True:
                try:
                    action, r, done = next(step) 
                    G = r + gamma*G
                    if done:
                        episode_returns[episode] = G 
                        episode += 1
                        G = 0

                    if visualize:
                        grid = (self.Q1 + self.Q2)/2
                        grid = np.max(grid, axis=1).reshape((self.env.height, self.env.width))
                        self.disp.process_events(grid, action)

                except StopIteration:
                    break
            avg_ep_returns += (episode_returns - avg_ep_returns)/(run+1)

        plt.plot(range(episodes),avg_ep_returns)
        plt.xlabel("Episode")
        plt.ylabel("Average Returns")
        plt.ylim([0,6])
        plt.show()

    def double_q(self, env, single_q=False):
        #self.Q1 = env.np_random.rand(env.width*env.height, len(env.actions))
        #self.Q2 = env.np_random.rand(env.width*env.height, len(env.actions))
        self.Q1 = np.zeros((env.width*env.height, len(env.actions)))
        self.Q2 = np.zeros((env.width*env.height, len(env.actions)))
        
        counts = np.ones((env.width*env.height, len(env.actions)))
        num_steps = 1

        for episode in range(episodes):
            done = False

            while not done:
                state = env.state
                action = get_eps_action(env, state, self.Q1+self.Q2)
                reward, done = env.step(env.actions[action])
                next_state = env.state

                yield env.actions[action], reward, done
                
                #reward += 1/counts[s2idx(state),action]#get_bonus(counts[s2idx(state),action], num_steps)
                step = (state,action,reward,next_state,done)
                
                if single_q:
                    double_q_update(self.Q1, self.Q1, env, step)
                elif env.np_random.uniform() < 0.5:
                    double_q_update(self.Q1, self.Q2, env, step)
                else:
                    double_q_update(self.Q2, self.Q1, env, step)

                num_steps += 1
                counts[s2idx(self.env, state), action] += 1

# converts environment state to Q index
def s2idx(env, state):
    return state[0]*env.width + state[1]

# adds exploration bonus to reward
def get_bonus(count, num_steps):
    return c*np.sqrt(np.log(num_steps)/count)

def add_bonus(Q1, Q2, num_steps, counts):
    exploration_bonus = c*np.sqrt(np.log(num_steps)/counts)
    Q1 = Q1 + exploration_bonus
    Q2 = Q2 + exploration_bonus
    return Q1, Q2


# update self.Q1 towards self.Q2 estimate of value of Q1's max action
def double_q_update(Q1, Q2, env, step):
    state, action, reward, next_state, done = step
    target = reward
    if not done:
        next_action = get_eps_action(env, next_state, Q1, 0) # eps=0 means pick argmax action
        target += gamma*Q2[s2idx(env, next_state), next_action]
    Q1[s2idx(env, state), action] += alpha*(target - Q1[s2idx(env, state), action])

# returns eps-greedy action
def get_eps_action(env, state, Q, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.randint(0, len(env.actions))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[s2idx(env, state)] == Q[s2idx(env, state)].max())) # to select argmax randomly
    return action

learner = Learner()
learner.main()
