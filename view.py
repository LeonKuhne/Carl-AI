import pygame
import numpy as np
import sys
from freenect import sync_get_depth as get_depth
import util

# config
config = util.read_config()

# make sure this is the same as util.py
NRESOLUTION = 2048

def make_gamma():
    """
    Create a gamma table
    """
    npf = float(NRESOLUTION)
    _gamma = np.empty((NRESOLUTION, 3), dtype=np.uint16)

    for i in range(NRESOLUTION):
        v = i / npf
        v = pow(v, 3) * 6
        pval = int(v * 6 * 256)
        lb = pval & 0xff
        pval >>= 8
        if pval == 0:
            a = np.array([255, 255 - lb, 255 - lb], dtype=np.uint8)
        elif pval == 1:
            a = np.array([255, lb, 0], dtype=np.uint8)
        elif pval == 2:
            a = np.array([255 - lb, lb, 0], dtype=np.uint8)
        elif pval == 3:
            a = np.array([255 - lb, 255, 0], dtype=np.uint8)
        elif pval == 4:
            a = np.array([0, 255 - lb, 255], dtype=np.uint8)
        elif pval == 5:
            a = np.array([0, 0, 255 - lb], dtype=np.uint8)
        else:
            a = np.array([0, 0, 0], dtype=np.uint8)

        _gamma[i] = a
    return _gamma

gamma = make_gamma()

fpsClock = pygame.time.Clock()
FPS = 30 # kinect only outputs 30 fps
disp_size = (640, 480)
pygame.init()
screen = pygame.display.set_mode(disp_size)
font = pygame.font.Font(pygame.font.get_default_font(), 32) # provide your own font 

while True:
    events = pygame.event.get()
    for e in events:
        if e.type == pygame.QUIT:
            sys.exit()
    fps_text = "FPS: {0:.2f}".format(fpsClock.get_fps())

    # get sensor data
    print(get_depth())
    depth = np.rot90(get_depth()[0]) # get the depth readings from the camera
        
    # capture the contour at 2100
    clip = config['clip']
    depth = np.clip(depth, clip['min'], clip['max']) # no way its bigger than 3000...

    # show the camera
    pixels = gamma[depth] # the colour pixels are the depth readings overlayed onto the gamma table
    temp_surface = pygame.Surface(disp_size)
    pygame.surfarray.blit_array(temp_surface, pixels)
    pygame.transform.scale(temp_surface, disp_size, screen)
    screen.blit(font.render(fps_text, 1, (255, 255, 255)), (30, 30))
    pygame.display.flip()
    fpsClock.tick(FPS)

