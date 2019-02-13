# base class for q-learning agents that prints, plots, and visualizes performance

import numpy as np
import matplotlib.pyplot as plt
from clamity_env import ClamEnv

alpha = 0.1
gamma = 1
eps = 0.3
c = 2
visualize = True
np.set_printoptions(linewidth=120)

class DoubleQLearner():
    def __init__(self, episodes, runs, trail=False, non_reward_trail=False):
        self.env = ClamEnv(trail=trail, non_reward_trail=non_reward_trail)
        self.step_taker = self.stepper
        self.episodes = episodes
        self.runs = runs
        self.trail = trail
        #if visualize:
        #    from display import Disp
        #    self.disp = Disp(self.env)


    def main(self):
        avg_ep_returns = np.zeros(self.episodes) # to plot
        num_visits = np.zeros((self.runs, self.env.height, self.env.width))    # times stopped in each state. measure of exploration
        run_returns = np.zeros(self.runs)   # average return for last 500 episodes of each run
        for run in range(self.runs):
            print(run)
            episode_returns = np.zeros(self.episodes)

            step = self.step_taker() # step generator. learning also happens
            G = 0
            episode = 0
            while True: # for each run
                try:
                    # needed for trail handling
                    state = self.env.state
                    if self.trail:
                        trail_penalty = self.env.comm_grid[state[0],state[1]]

                    # take a step
                    obs, action, r_ext, r_int, next_obs, done = next(step) 

                    # update visit counts
                    if action=="stop":
                        num_visits[run,state[0],state[1]] += 1

                    # correcting for the self-imposed negative rewards (trail) for comparison purposes
                    if self.trail and action=="stop":
                        r_ext -= trail_penalty

                    G = r_ext + gamma*G
                    if done:
                        episode_returns[episode] = G    # record return for each episode
                        episode += 1
                        G = 0
                    if visualize and episode%100 == 99 and done:   # plays animation of agent during learning. very slow
                        self.visualizer(episode, obs, action, next_obs)
                    if episode%100 == 99 and done:
                        print("episode: ", episode+1)
                        print("avg return: ", np.average(episode_returns[episode-100:episode]))
                except StopIteration:   # happens when the step generator completes all episodes for this run
                    break
            avg_ep_returns += (episode_returns - avg_ep_returns)/(run+1)    # update line to plot
            run_returns[run] = np.average(episode_returns[-500:])   # save run return data
            
        # plot average returns per episode
        plt.plot(range(self.episodes),avg_ep_returns)
        plt.xlabel("Episode")
        plt.ylabel("Average Returns")
        #plt.ylim([0,1])
        plt.savefig('rnd_non_reward_trail.png')
        plt.show()

        # print the average return over the last 500 episodes
        print(np.average(avg_ep_returns[-500:]))

        # save things to files
        with open("num_visits_rnd.npy", 'wb') as f:
            pickle.dump(num_visits, f)
        with open("run_returns_rnd.npy", 'wb') as f:
            pickle.dump(run_returns, f)

    def visualizer(self, episode, obs, action, next_obs):
        grid = np.zeros((self.env.height, self.env.width, self.num_actions, 2))

        for row in range(self.env.height):
            for col in range(self.env.width):
                grid_point = [row, col]
                Q_ext, Q_int = self.sess.run([self.Q_ext, self.Q_int], {self.inputs_ph: self.env.obs_fn(grid_point)})
                grid[row,col,:,0] = Q_ext
                grid[row,col,:,1] = Q_int
        
        # for tabular double q learning
        #grid = (self.Q1 + self.Q2)/2
        #grid = np.max(grid, axis=1).reshape((self.env.height, self.env.width)) # assigning a value to each state
        
        #print(np.max(grid[:,:,:,0], 2)) # max values for Q_ext
        print(np.ceil(np.average(grid[:,:,:,1], 2))) # average Q_int values. ceil is for readability
        #grid = np.max(grid[...,0]+grid[...,1],2) # max values for Q_ext+Q_int

        #self.disp.process_events(grid, state, action, next_state)
        #print("ep:"+str(episode)+action+" state:"+str(state)+" next_state:"+str(next_state))


    def stepper(self):
        raise NotImplementedError
