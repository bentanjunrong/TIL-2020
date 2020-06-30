import png
import cv2
from Tello_api import *

THRESHOLD = 80                              # black threshold
LENGTH = 9                                  # arena length
HEIGHT = 3                                  # arena height
GRID = [[-1]*LENGTH for i in range(HEIGHT)] # occupancy grid
ROW, COL = 0, 0                             # current grid position

def calc(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    res = 0
    for row in img:
        res += sum(row)
    return res / sum([len(row) for row in img])

def append(val):
    global COL, ROW, GRID

    GRID[ROW % HEIGHT][COL % LENGTH] = 1 if val > THRESHOLD else 0
    COL += 1
    ROW = ROW + 1 if not COL % LENGTH else ROW

# load drone
tello = Tello()
tello.startvideo()
tello.startstates() # to read height of tello
cv2.namedWindow('Tello video', cv2.WINDOW_NORMAL)

while tello.frame is None: # this is for video warm up. when frame is received, this loop is exited.
    pass

last_val = None
while True:
    cv2.imshow('Tello video', cv2.flip(tello.frame, 0)) # flip required; using mirror

    k = cv2.waitKey(16)

    if k == ord('a'):
        # append to GRID
        append(last_val)

    elif k == ord('p'):
        # calc whiteness
        print('taking photo...')
        last_val = calc(tello.frame)
        print('whiteness is', last_val, '..........0 (pitch black) and 255 (bright white)')
        print('photo taken and analysed successfully.')

    elif k == ord('r'):
        # r revert last, if exists
        print('reverting last photo...')
        GRID[ROW % HEIGHT][COL % LENGTH] = -1
        ROW = ROW - 1 if ROW and not COL % LENGTH else ROW
        COL = COL - 1 if COL else COL
        print('last photo reverted.')

    elif k == 27: # press esc to stop
        tello.exit()
        break

    elif k != -1: # press wasdqe to control tello. u,j to control height. t,l to takeoff and land
        tello.act(k) 
