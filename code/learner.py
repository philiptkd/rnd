import numpy as np
import matplotlib.pyplot as plt
from clamity_env import ClamEnv

alpha = 0.1
gamma = 1
eps = 0.3
c = 2
visualize = False

class DoubleQLearner():
    def __init__(self, episodes, runs):
        self.env = ClamEnv()
        self.step_taker = self.stepper
        self.episodes = episodes
        self.runs = runs
        if visualize:
            from display import Disp
            self.disp = Disp(self.env)


    def main(self):
        avg_ep_returns = np.zeros(self.episodes) # to plot
        for run in range(self.runs):
            print(run)
            episode_returns = np.zeros(self.episodes)

            step = self.step_taker() # step generator. learning also happens
            G = 0
            episode = 0
            while True:
                try:
                    action, r, done = next(step) 
                    G = r + gamma*G
                    if done:
                        episode_returns[episode] = G    # record return for each episode
                        episode += 1
                        G = 0

                    if visualize:   # plays animation of agent during learning. very slow
                        grid = (self.Q1 + self.Q2)/2
                        grid = np.max(grid, axis=1).reshape((self.env.height, self.env.width)) # assigning a value to each state
                        self.disp.process_events(grid, action)

                except StopIteration:
                    break
            avg_ep_returns += (episode_returns - avg_ep_returns)/(run+1)

        plt.plot(range(self.episodes),avg_ep_returns)
        plt.xlabel("Episode")
        plt.ylabel("Average Returns")
        #plt.ylim([0,1])
        plt.show()

    def stepper(self):
        raise NotImplementedError
