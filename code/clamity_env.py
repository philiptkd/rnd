# a simple Clamity-like environment
# discrete state and action spaces
# the agent receives 0 reward until it decides to stop
    # then it receives however much reward is marked on the grid at that spot
    # stopping ends the episode

import numpy as np

class ClamEnv():
    def __init__(self):
        # grid setup
        self.height = 5
        self.width = 5
        self.start = (self.height//2,self.width//2)
        self.grid = np.zeros((self.height, self.width))
        self.grid[1,1] = 2
        self.grid[1,self.width-2] = 3
        self.grid[self.height-2,1] = 4
        self.grid[self.height-2,self.width-2] = 5
        self.grid[self.start[0], self.start[1]] = 1
        
        self.actions = ("left","right","up","down","stop")
        self.np_random = np.random.RandomState()
        self.reset()

    # resets to start position
    def reset(self):
        self.state = list(self.start)

    # returns (reward, done)
    def step(self, action):
        row, col = self.state

        # transition to next state
        if action == "stop":
            reward = self.grid[row, col]
            done = True
            self.reset()
        else:
            if action == "left":
                self.state = [row, max(0, col-1)]
            elif action == "right":
                self.state = [row, min(self.width-1, col+1)]
            elif action == "up":
                self.state = [max(0, row-1), col]
            else: # if action == "down":
                self.state = [min(self.height-1, row+1), col]
            reward = 0
            done = False

        return reward, done
