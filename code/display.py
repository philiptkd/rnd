import pygame, sys
from pygame.locals import *

# constants 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
XAXIS = 0
YAXIS = 1
tilesize = 40
radius = tilesize//4
delay = 100

class Disp():
    def __init__(self, env):
        # parameters
        self.map_width = env.width
        self.map_height = env.height
        self.player_color = BLACK

        # create a new display surface
        pygame.init()
        self.surface = pygame.display.set_mode((self.map_width*tilesize, self.map_height*tilesize))
        pygame.display.set_caption('My First Game')
        self.fpsClock = pygame.time.Clock()

        # initialize drawings
        pygame.draw.rect(self.surface, WHITE, (0,0,self.map_width*tilesize,self.map_height*tilesize))
        pygame.draw.circle(self.surface, self.player_color, g2c(env.start[1], env.start[0]), radius)
        pygame.display.update()
        
        

    # for coloring the grid by Q values
    def draw_surface(self, Q):
        for i in range(self.map_width): # x/col
            for j in range(self.map_height):    # y/row
                value = 255 - int(Q[j,i]*255/5) # the max value with no exploration bonus is 5
                value = max(value, 0) # exploration bonus will cause Q > 5
                color = (value,value,255)
                try:
                    pygame.draw.rect(self.surface, color, (i*tilesize,j*tilesize,tilesize,tilesize))    # (left,top,width,height)
                except:
                    print(color)    # debugging
                    raise Exception

    # animates a transition between steps by linearly interpolating
    def animate(self, state, next_state, Q):
        start_pos = g2c(state[1], state[0])
        end_pos = g2c(next_state[1], next_state[0])
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        for i in range(tilesize):
            x = int(start_pos[0] + (i/tilesize)*dx)
            y = int(start_pos[1] + (i/tilesize)*dy)

            # draw things
            self.draw_surface(Q)
            pygame.draw.circle(self.surface, self.player_color, (x,y), radius)
        
            # update display
            pygame.display.update()
            self.fpsClock.tick(delay)

    def process_events(self, Q, state, action, next_state):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # animate the last action and update Q
        # if we just stopped, change the player color to green
        if action == "stop":
            self.player_color = GREEN
            self.animate(state, state, Q)  # displays Q without changing position

            # change color back to black and reset to starting position
            self.player_color = BLACK
            self.animate(next_state, next_state, Q)
        
        else:   # animate movement between states
            self.animate(state, next_state, Q)


# converts grid coords to surface coords for center of circle
def g2c(x,y):
    return (int(tilesize*(x+.5)),int(tilesize*(y+.5)))
