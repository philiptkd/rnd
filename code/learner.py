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
            while True: # for each run
                try:
                    state, action, r_ext, r_int, next_state, done = next(step) 
                    G = r_ext + gamma*G
                    if done:
                        episode_returns[episode] = G    # record return for each episode
                        episode += 1
                        G = 0
                    if visualize and episode%100 == 99:   # plays animation of agent during learning. very slow
                        self.visualizer(episode, state, action, next_state)
                    if episode%100 == 99:
                        print("avg return: ", np.average(episode_returns[episode-100:episode]))
                except StopIteration:   # happens when the step generator completes all episodes for this run
                    break
            avg_ep_returns += (episode_returns - avg_ep_returns)/(run+1)
            
        plt.plot(range(self.episodes),avg_ep_returns)
        plt.xlabel("Episode")
        plt.ylabel("Average Returns")
        #plt.ylim([0,1])
        plt.show()
        #plt.savefig('latest_neural_fig.png')
        print(np.average(avg_ep_returns[-500:]))


    def visualizer(self, episode, state, action, next_state):
        grid = np.zeros((self.env.height, self.env.width, self.num_actions, 2))

        for row in range(self.env.height):
            for col in range(self.env.width):
                grid_point = [row, col]
                Q_ext, Q_int = self.sess.run([self.Q_ext_out_op, self.Q_int_out_op], {self.inputs_ph: self.one_hot(grid_point)})
                grid[row,col,:,0] = Q_ext
                grid[row,col,:,1] = Q_int
        
        # for tabular double q learning
        #grid = (self.Q1 + self.Q2)/2
        #grid = np.max(grid, axis=1).reshape((self.env.height, self.env.width)) # assigning a value to each state
        
        #print(np.max(grid[:,:,:,0], 2)) # max values for Q_ext
        print(grid[:,:,:,1]) # max values for Q_int
        grid = np.max(grid[...,0]+grid[...,1],2) # max values for Q_ext+Q_int

        self.disp.process_events(grid, state, action, next_state)
        #print("ep:"+str(episode)+action+" state:"+str(state)+" next_state:"+str(next_state))


    def stepper(self):
        raise NotImplementedError
