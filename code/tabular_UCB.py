import numpy as np
import matplotlib.pyplot as plt
from clamity_env import ClamEnv

alpha = 0.1
gamma = 1
eps = 0.3
episodes = 2000
runs = 20
c = 2

def main(env, step_taker):
    avg_ep_returns = np.zeros(episodes) # to plot
    for run in range(runs):
        print(run)
        episode_returns = np.zeros(episodes)

        step = step_taker(env) # step generator. learning also happens
        G = 0
        episode = 0
        while True:
            try:
                r,done = next(step) 
                G = r + gamma*G
                if done:
                    episode_returns[episode] = G 
                    episode += 1
                    G = 0
            except StopIteration:
                break
        avg_ep_returns += (episode_returns - avg_ep_returns)/(run+1)

    plt.plot(range(episodes),avg_ep_returns)
    plt.xlabel("Episode")
    plt.ylabel("Average Returns")
    plt.ylim([0,5])
    plt.show()

def double_q(env, single_q=False):
    Q1 = env.np_random.rand(env.width*env.height, len(env.actions))
    Q2 = env.np_random.rand(env.width*env.height, len(env.actions))
    counts = np.ones((env.width*env.height, len(env.actions)))
    num_steps = 1

    for episode in range(episodes):
        done = False

        while not done:
            state = env.state
            action = get_eps_action(env, state, Q1+Q2)
            reward, done = env.step(env.actions[action])
            next_state = env.state
            
            yield reward, done
            
            #reward += 1/counts[s2idx(state),action]#get_bonus(counts[s2idx(state),action], num_steps)
            step = (state,action,reward,next_state,done)
            
            if single_q:
                double_q_update(Q1,Q1,env,step)
            elif env.np_random.uniform() < 0.5:
                double_q_update(Q1,Q2,env,step)
            else:
                double_q_update(Q2,Q1,env,step)

            num_steps += 1
            counts[s2idx(state), action] += 1

# adds exploration bonus to reward
def get_bonus(count, num_steps):
    return c*np.sqrt(np.log(num_steps)/count)

def add_bonus(Q1,Q2,num_steps,counts):
    exploration_bonus = c*np.sqrt(np.log(num_steps)/counts)
    q1 = Q1 + exploration_bonus
    q2 = Q2 + exploration_bonus
    return q1,q2

# converts environment state [a,b] to state index a*b
def s2idx(state):
    return state[0]*state[1]

# update Q1 towards Q2 estimate of value of Q1's max action
def double_q_update(Q1, Q2, env, step):
    state, action, reward, next_state, done = step
    target = reward
    if not done:
        next_action = get_eps_action(env, next_state, Q1, 0) # eps=0 means pick argmax action
        target += gamma*Q2[s2idx(next_state), next_action]
    Q1[s2idx(state), action] += alpha*(target - Q1[s2idx(state), action])

# returns eps-greedy action
def get_eps_action(env, state, Q, eps=eps):
    if env.np_random.uniform() < eps:
        action = env.np_random.randint(0, len(env.actions))
    else:
        action = env.np_random.choice(np.flatnonzero(Q[s2idx(state)] == Q[s2idx(state)].max())) # to select argmax randomly
    return action

main(ClamEnv(), double_q)
