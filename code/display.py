import pygame, sys
from pygame.locals import *

# constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
TILESIZE = 40
MAPWIDTH = 9 # use grid width
MAPHEIGHT = 9 # use grid height
XAXIS = 0
YAXIS = 1
RADIUS = TILESIZE//2
DELAY = 150

# converts grid coords to surface coords for center of circle
def g2c(x,y):
    return (int(TILESIZE*(x+.5)),int(TILESIZE*(y+.5)))

# animates a step in a cardinal direction
def animate(old_pos, axis, sign):
    loc = list(g2c(*old_pos))
    for i in range(TILESIZE):
        # update circle location
        loc[axis] += sign   # move +/- 1 on given axis

        # draw things
        pygame.draw.rect(DISPLAYSURF, WHITE, (0,0,MAPWIDTH*TILESIZE,MAPHEIGHT*TILESIZE))
        pygame.draw.circle(DISPLAYSURF, BLUE, loc, RADIUS)
    
        # update display
        pygame.display.update()
        fpsClock.tick(DELAY)

    new_pos = old_pos
    new_pos[axis] += sign
    return new_pos


# create a new display surface
pygame.init()
DISPLAYSURF = pygame.display.set_mode((MAPWIDTH*TILESIZE, MAPHEIGHT*TILESIZE))
pygame.display.set_caption('My First Game')
fpsClock = pygame.time.Clock()

# initialize drawings
pos = [0,0]
pygame.draw.rect(DISPLAYSURF, WHITE, (0,0,MAPWIDTH*TILESIZE,MAPHEIGHT*TILESIZE))
pygame.draw.circle(DISPLAYSURF, BLUE, g2c(*pos), RADIUS)
pygame.display.update()

# wait for events
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_RIGHT:
                pos = animate(pos, XAXIS, 1)
            elif event.key == K_LEFT:
                pos = animate(pos, XAXIS, -1)
            elif event.key == K_UP:
                pos = animate(pos, YAXIS, -1)
            elif event.key == K_DOWN:
                pos = animate(pos, YAXIS, 1)
