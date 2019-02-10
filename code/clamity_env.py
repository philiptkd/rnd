# a simple Clamity-like environment
# discrete state and action spaces
# the agent receives 0 reward until it decides to stop
    # then it receives however much reward is marked on the grid at that spot
    # stopping ends the episode

import numpy as np

class ActionSpace():
    def __init__(self, actions_tuple):
        self.actions = actions_tuple
        self.np_random = np.random.RandomState()
    
    def sample(self):
        action = self.np_random.randint(0, len(self.actions))
        return action



class ClamEnv():
    def __init__(self):
        # grid setup
        self.height = 3
        self.width = 3
        self.start = (self.height//2,self.width//2)
        self.grid = np.zeros((self.height, self.width))
        self.grid[0,0] = 2
        self.grid[0,self.width-1] = 3
        self.grid[self.height-1,0] = 4
        self.grid[self.height-1,self.width-1] = 5
        self.grid[self.start[0], self.start[1]] = 1
        
        self.action_space = ActionSpace(("left","right","up","down","stop"))
        self.reset()

    # resets to start position
    def reset(self):
        self.state = list(self.start)
        return self.list2int(self.state)

    # action is an index into possible action tuple
    # returns (reward, done)
    def step(self, action):
        row, col = self.state

        # transition to next state
        if self.action_space.actions[action] == "stop":
            reward = self.grid[row, col]
            done = True
            self.reset()
        else:
            if self.action_space.actions[action] == "left":
                self.state = [row, max(0, col-1)]
            elif self.action_space.actions[action] == "right":
                self.state = [row, min(self.width-1, col+1)]
            elif self.action_space.actions[action] == "up":
                self.state = [max(0, row-1), col]
            else: # if action == "down":
                self.state = [min(self.height-1, row+1), col]
            reward = 0
            done = False

        return self.state, reward, done, {} # openai gym convention

    # converts state list to int
    def list2int(self, state):
        return state[0]*self.width + state[1]
