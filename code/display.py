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
        self.start = (env.start[1], env.start[0])   # env is in (row,col) but we want (x,y)
        self.player_color = BLACK

        # create a new display surface
        pygame.init()
        self.surface = pygame.display.set_mode((self.map_width*tilesize, self.map_height*tilesize))
        pygame.display.set_caption('My First Game')
        self.fpsClock = pygame.time.Clock()

        # initialize drawings
        self.pos = list(self.start)
        pygame.draw.rect(self.surface, WHITE, (0,0,self.map_width*tilesize,self.map_height*tilesize))
        pygame.draw.circle(self.surface, self.player_color, self.g2c(*self.pos), radius)
        pygame.display.update()
        
        
    # converts grid coords to surface coords for center of circle
    def g2c(self, x,y):
        return (int(tilesize*(x+.5)),int(tilesize*(y+.5)))

    # for coloring the grid by Q values
    def draw_surface(self, Q):
        for i in range(self.map_width): # x/col
            for j in range(self.map_height):    # y/row
                value = 255 - int(Q[j,i]*255/5) # the max value with no exploration bonus is 5
                color = (value,value,255)
                pygame.draw.rect(self.surface, color, (i*tilesize,j*tilesize,tilesize,tilesize))    # (left,top,width,height)

    # animates a step in a cardinal direction
    def animate(self, old_pos, axis, sign, Q):
        loc = list(self.g2c(*old_pos))
        for i in range(tilesize):
            # update circle location
            loc[axis] += sign   # move +/- 1 on given axis

            # draw things
            self.draw_surface(Q)
            pygame.draw.circle(self.surface, self.player_color, loc, radius)
        
            # update display
            pygame.display.update()
            self.fpsClock.tick(delay)

        new_pos = old_pos
        new_pos[axis] += sign
        return new_pos

    def process_events(self, Q, action):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        
        # animate the last action and update Q
        if action == "right" and self.pos[0] < self.map_width-1:
            self.pos = self.animate(self.pos, XAXIS, 1, Q)
        elif action == "left" and self.pos[0] > 0:
            self.pos = self.animate(self.pos, XAXIS, -1, Q)
        elif action == "up" and self.pos[1] > 0:
            self.pos = self.animate(self.pos, YAXIS, -1, Q)
        elif action == "down" and self.pos[1] < self.map_height-1:
            self.pos = self.animate(self.pos, YAXIS, 1, Q)
            # if we just stopped, change the player color to green
        elif action == "stop":
            self.player_color = GREEN
            self.pos = self.animate(self.pos, XAXIS, 0, Q)  # displays Q without changing position

            # change color back to black and reset to starting position
            self.player_color = BLACK
            self.pos = list(self.start)
            self.pos = self.animate(self.pos, XAXIS, 0, Q)
        else:   # it ran into a wall
            self.pos = self.animate(self.pos, XAXIS, 0, Q)  # displays Q without changing position

