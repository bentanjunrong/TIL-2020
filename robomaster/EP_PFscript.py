from EP_api import Robot, findrobotIP
from utils.shortest_path import find_shortest_path
import time
import cv2
import numpy



# activate ONLY when using robot

# try:
#         robot = Robot(findrobotIP())
# except:
#     robot = False

# Pathfinding script initialize values (grid related values might need to be changed)

start_disp = 0.2 # the robot is roughly 20cm south of the center of the junction it is facing when it starts. Untested.
start_angle = 0 # robot starts facing the search and rescue area
end_angle = 0 # robot has to end facing the search and rescue area
start_loc = (2,6) # grid coordinates of the Start marker 
end_loc = (0,6) # grid coordinates of the End marker
angle_map = { # for turn_to_next
    (-1,0): 0,
    (0,1): 90,
    (1,0): 180,
    (0,-1): -90
}



def waitToStill():
    time.sleep(0.5)
    still = False
    while (still is False):
        state = robot._sendcommand('chassis status ?', echo=False).split(' ')
        # print (state)
        if state[0] == 'ok':
            continue
        elif int(state[0]) == 1:
            still = True
    return

def moveforward(dist):
    robot.move('x {} vxy 0.1'.format(str(dist)))
    waitToStill()

def get_current_angle():
    current_angle = float(robot._sendcommand('chassis position ?').split(' ')[2]) + start_angle # Relative to power-up/starting position, defined in start_angle
    if current_angle > 180: current_angle -= 360
    return current_angle

def turn_to_angle(final_angle):
    current_angle = get_current_angle()
    angle_to_turn = final_angle - current_angle
    robot.rotate(str(angle_to_turn))
    waitToStill()


def turn_to_next(current,next):
    # delta_row = next[0] - current[0]
    # delta_col =  next[1] - current[1] 
    final_angle = angle_map[tuple(numpy.subtract(next, current))] # Angle that robot must turn to.
    turn_to_angle(final_angle)    

def check_depth(current_loc,path): # Tested
    next_loc = path[-1]
    dir_vector = tuple(numpy.subtract(next_loc, current_loc))
    depth = 1
    while True:
        peek_loc = path[-2]
        if dir_vector == tuple(numpy.subtract(peek_loc,next_loc)): # if next segment is also along same forward axis
            depth += 1
            next_loc = peek_loc
            path.pop()
        else: break
    return depth, path

def nearest_90_angle():
    angle = get_current_angle()
    if -135 < angle < -45: return -90
    elif -45 < angle < 45: return 0
    elif 45 < angle < 135: return 90
    else: return 180

# Aligns robot so that it is in the center of the current_loc


def align(current_loc):
    dir_vector = tuple(numpy.subtract(current_loc, start_loc))
    junct_sep = 0.8 # distance between center of two junctions
    x_disp = start_disp + abs(dir_vector[0])*junct_sep
    y_disp = dir_vector[1]*junct_sep # no absolute here because y can be negative or positive
    snap_angle = nearest_90_angle() # finds nearest 90 degree angle. I call it snapping lmao
    turn_to_angle(snap_angle)
    x_robot, y_robot = float(robot._sendcommand('chassis position ?').split(' ')[0:1])
    if snap_angle == 0:
        x = x_disp - x_robot
        y = y_disp - y_robot
    elif snap_angle == 90:
        x = y_disp - y_robot
        y = x_robot - x_disp
    elif snap_angle == -90:
        x = y_robot - y_disp
        y = x_disp - x_robot
    else:
        x = x_robot - x_disp
        y = y_robot - y_disp
    robot.move('x {} y {} vxy 0.1'.format(str(x), str(y)))
    waitToStill()


    
   

# Main Function. Accepts occupancy grid (should be 9x3, but u can test with other sizes. Must be of 1s and 0s ONLY)
def navigate_start_to_end(grid):
    if robot is False: return False
    
    path = find_shortest_path(start_loc,end_loc,grid)
    
    moveforward(start_disp) # enter into first junction (untested distance)
    current_loc = path.pop()
    while current_loc is not end_loc: # navigate to segment where the end marker is next to.
        turn_to_next(current_loc,path[-1]) # checks if robot needs to face another direction before moving forward
        depth, trimmed_path = check_depth(current_loc,path) # check how many segments robot needs to move forward (should be multiples of 2) and the trimmed path (removes all segments before the ending segment)
        path = trimmed_path
        moveforward(0.2*(depth/2)) # value here refers to distance between centre of one junction to another. 0.2 is untested.        
        current_loc = path.pop()
        align(current_loc) # ensures robot is centered to current junction before calculating next movement
    turn_to_angle(end_angle)
    moveforward(start_disp)


