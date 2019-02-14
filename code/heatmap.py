import pickle
import pygame
from pygame.locals import *
import numpy as np
import sys

tilesize = 40

def load(filename):
    with open(filename, 'rb') as f:
        counts = pickle.load(f)
    return counts

def disp(arr, max_val, filename):
    max_val = np.log(max_val)
    for x in range(width):
        for y in range(height):
            value = 255 - int(np.log(arr[y,x])*255/max_val)
            color = (255, value, value)
            pygame.draw.rect(surface, color, (x*tilesize,y*tilesize,tilesize,tilesize))
    pygame.display.update()
    pygame.image.save(surface, filename)

            
f1 = './partial_neural_feature_trail_15x15/num_visits.npy'
f2 = './partial_neural_reward_trail_15x15/num_visits.npy'
f3 = './partial_rnd_feature_trail_15x15/num_visits.npy'
f4 = './partial_rnd_reward_trail_15x15/num_visits.npy'

# sum over run dimension to get counts over entire experiment
arr1 = np.sum(load(f1), 0)
arr2 = np.sum(load(f2), 0)
arr3 = np.sum(load(f3), 0)
arr4 = np.sum(load(f4), 0)

# for scaling purposes
max_count = np.max(np.stack([arr1, arr2, arr3, arr4]))

# initialize pygame
pygame.init()
height, width = arr1.shape
surface = pygame.display.set_mode((width*tilesize, height*tilesize))

for i,arr in enumerate([arr1, arr2, arr3, arr4]):
    disp(arr, max_count, str(i)+".png")
