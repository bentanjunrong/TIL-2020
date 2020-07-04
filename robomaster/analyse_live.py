import png
import cv2
import json
from utils.Tello_api import *
from EP_PFscript import navigate_start_to_end
from map_to_path.occupancy_map_8n import plot_path

THRESHOLD = 80                              # black threshold
LENGTH = 9                                  # arena length
HEIGHT = 3                                  # arena height
# GRID = [[-1]*LENGTH for i in range(HEIGHT)] # occupancy grid
# ROW, COL = 0, 0                             # current grid position

with open('robomaster/listfile.txt', 'r') as filehandle:
    temp = json.load(filehandle)
    GRID, ROW, COL = temp[0], temp[1], temp[2]

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
    with open('robomaster/listfile.txt', 'w') as filehandle:
        json.dump([GRID, ROW, COL], filehandle)

# load drone
tello = Tello()
tello.startvideo()
tello.startstates() # to read height of tello
cv2.namedWindow('Tello Live Video', cv2.WINDOW_NORMAL)
cv2.namedWindow('Tello Last Image Taken', cv2.WINDOW_NORMAL)

while tello.frame is None: # this is for video warm up. when frame is received, this loop is exited.
    pass

last_val = None
last_frame = None
while True:
    cv2.imshow('Tello Live Video', cv2.flip(tello.frame, 0)) # flip required; using mirror

    k = cv2.waitKey(16)

    if k == ord('r'):
        # append to GRID
        if last_val == None: print("Photo has not been taken yet!")
        else: append(last_val)

    elif k == ord('p'): # every pic print the grid
        # calc whiteness
        print('taking photo...')
        last_frame = tello.frame
        last_val = calc(last_frame)
        cv2.imshow('Tello Last Image Taken', cv2.flip(last_frame, 0)) # flip required; using mirror
        print('whiteness is', last_val, '..........0 (pitch black) and 255 (bright white)')
        tmp = 1 if last_val > THRESHOLD else 0
        print('appending val is ', tmp)

    elif k == ord('r'):
        # r revert last, if exists
        print('reverting last photo...')
        ROW = ROW - 1 if ROW and not COL % LENGTH else ROW
        COL = COL - 1 if COL else COL
        GRID[ROW % HEIGHT][COL % LENGTH] = -1
        print('last photo reverted.')

    elif k == ord('o'):
        # append 1
        append(999)

    elif k == ord('z'):
        # append 0
        append(0)

    elif k == 'x':
        # load last image to occupancy modeland imwrite to com the res
        if last_frame == None: print("Photo has not been taken yet!")
        else:
            print("Plotting path on last frame taken....")
            plot_path(last_frame)

    elif k == 'g':
        # view grid
        print(GRID)

    elif k == 27:
        # press esc to stop
        tello.exit()
        print('Tello landing....')
        print('Passing GRID to RoboMaster....')
        # convert GRID if drone used snake pattern
        res = []
        for i in range(len(GRID)):
            res.append(GRID[i][::-1] if i % 2 else GRID[i])
        navigate_start_to_end(res)
        break

    elif k != -1: # press wasdqe to control tello. u,j to control height. t,l to takeoff and land
        tello.act(k) 
