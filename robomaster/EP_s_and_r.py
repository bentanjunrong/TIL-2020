from EP_api import Robot, findrobotIP
import time
import cv2
import numpy


robot = Robot(findrobotIP)
target_cat = 0
target_coords = []



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
    # return #TEST

def moveforward(dist): # tested
    robot.move('x {} vxy 0.1'.format(str(dist)))
    waitToStill()

def get_current_angle():
    current_angle = float(robot._sendcommand('chassis position ?').split(' ')[2]) + start_angle # Relative to power-up/starting position, defined in start_angle
    if current_angle > 180: current_angle -= 360
    return current_angle
    # return 90 #TEST

def turn_to_angle(final_angle): #tested
    current_angle = get_current_angle()
    angle_to_turn = final_angle - current_angle
    robot.rotate(str(angle_to_turn))
    waitToStill()

def nearest_90_angle(): # tested
    angle = get_current_angle()
    if -135 < angle < -45: return -90
    elif -45 <= angle <= 45: return 0
    elif 45 < angle < 135: return 90
    else: return 180



# Aligns robot so that it is in the center of the current_loc

# coords is an float list sized 3 containing x,y,z coords of the destination
def align(coords): #Tested (notes above)
    x_dest, y_dest, z_dest = coords
    turn_to_angle(z_dest)
    snap_angle = nearest_90_angle() # finds nearest 90 degree angle. I call it snapping lmao
    turn_to_angle(snap_angle)
    x_cur, y_cur = robot._sendcommand('chassis position ?').split(' ')[:2]
    x_robot = float(x_robot)
    y_robot = float(y_robot)
    print(x_robot,y_robot)
    # x_robot, y_robot = (0.2,-0.6) # TEST
    if snap_angle == 0:
        x = x_dest - x_robot
        y = y_dest - y_robot
    elif snap_angle == 90:
        x = y_dest - y_robot
        y = x_robot - x_dest
    elif snap_angle == -90:
        x = y_robot - y_dest
        y = x_dest - x_robot
    else:
        x = x_robot - x_dest
        y = y_robot - y_dest
    robot.move('x {} y {} vxy 0.1'.format(str(x), str(y)))
    waitToStill()

def go_center():
    align([0,0,0])

def save_coords():
    target_coords = robot._sendcommand('chassis position ?').split(' ')[:3]

def flash_green():
    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect blink')
    time.sleep(5)

def flash_red():
    robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect blink')
    time.sleep(5)
tagged = [False*3]
tag_count = 0
turn_const = 5 # UNTESTED
dist_thresh = 0.5 # UNTESTED


def search_loop(result):
    detected = result['detect']
    dist = result['dist']
    cat = result['class']
    if detected is False:
        robot.rotate('5')
        waitToStill()
        return False
    elif dist > dist_thresh or dist < -dist_thresh:
        turn_angle = dist*turn_const # if dist is negative, will turn left
        robot.rotate(str(turn_angle))
        waitToStill()
        return False
    elif cat == 0:
        moveforward(0.3)
        return False
    else:
        if cat == target_cat:
            save_coords()
            flash_green()
        else:
            flash_red()
        tagged[tag_count]  = True
        tag_count += 1
        
        if tag_count == 3:
            return True
        else: 
            go_center()
            if tag_count == 1: robot.rotate('-45')
            else : robot.rotate('45')
            waitToStill()
            return False


def s_and_r(target)
    target_cat = target_cat
    robot.rotate('-135')
    robot.startvideo()
    # Search loop
    while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
        pass

    frame_counter = 0
    search_completed = False
    pickup_completed = False
    while True:
        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        tmp_frame = robot.frame
        cv2.imshow('Live video', tmp_frame) # access the video feed by robot.frame
        if counter % 30 == 0:
            result = predict_frame(tmp_frame) # RAPHAEL'S FUNCTION GOES HERE
            if search_completed is False:
                search_completed = search_loop(result)
            elif pickup_completed is False:
                # pickup_completed = pickup_loop(result)
                robot.exit()
                break
            else:
                robot.exit()
                break
                
            
        k = cv2.waitKey(16) & 0xFF
        if k == 27: # press esc to stop
            print("Quitting")
            robot.exit()
            break