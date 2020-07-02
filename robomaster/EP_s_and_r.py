from EP_api import Robot, findrobotIP
import time
import cv2
import numpy
from vidtobb import detect_object


robot = Robot(findrobotIP)
target_cats = 0
target_coords = []
center_coords = []
partials_coords = []
grip_thresh = 0


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
    current_angle = float(robot._sendcommand('chassis position ?').split(' ')[2]) # Relative to power-up/starting position, which should be facing north
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
    robot.move('x {} y {} z {} vxy 0.1'.format(str(x), str(y),str(z_dest)))
    waitToStill()
def crop_frame_by(frame,crop_by_factor):
    crop_by = crop_by_factor #lol pls forgive me for this
    width = (1280/crop_by)/2
    x_left = int((640) - width) # from center
    x_right = int((640) + width) #from center
    return frame[0:720,x_left:x_right]



def rescue_grip(): # Tested
    robot._sendcommand('robotic_arm moveto x 182 y 0')

def calc_image(): # Tested
    img = cv2.cvtColor(robot.frame,cv2.COLOR_BGR2GRAY)[549:625, 603:683]
    res = 0
    for row in img:
        res += sum(row)
    return res / sum([len(row) for row in img])

def set_grip_threshold(): # Tested
    global grip_thresh
    rescue_grip()
    time.sleep(2)
    robot.closearm()
    time.sleep(3)
    grip_thresh = calc_image()
    time.sleep(3)
    robot.openarm()
    robot.center() 
    time.sleep(1)
    print("GRIP THRESHOLD: ",grip_thresh)
    return True


def is_gripped(): # Tested
    # val = int(robot._sendcommand('robotic_gripper status ?'))
    # if val == 1: return True
    # else: return False
    val = calc_image()
    print('CURRENT THRESHOLD: ',val)
    if val > (grip_thresh + 3) or val < (grip_thresh - 3): return True
    else: return False

def save_target_coords():
    target_coords = robot._sendcommand('chassis position ?').split(' ')[:3]

def save_center_coords():
    center_coords = robot._sendcommand('chassis position ?').split(' ')[:3]

def flash_green():
    robot._sendcommand('led control comp bottom_all r 0 g 255 b 0 effect blink')
    time.sleep(5)

def flash_red():
    robot._sendcommand('led control comp bottom_all r 255 g 0 b 0 effect blink')
    time.sleep(5)


binary_lock = False # when an object is detected, set to True so that if subsequently there are no detections, the robot doesn't try to rotate. Will be reset once tagging is complete

def lock_on_loop(result):
    if result: # doll detected
        binary_lock = True
    else: 
        robot.rotate('5') # assuming left-to-right sweep
        waitToStill()
def is_full_match(cat):
    matches = []
    for target_cat in target_cats:
        if target_cat in cat:
            matches.append(target_cat)
    if len(matches) == len(target_cats):
        return True
    
    elif len(matches): 
        partials_coords.append(robot._sendcommand('chassis position ?').split(' ')[:3])
    return False

        



tag_count = 0
turn_const = 0.01 # UNTESTED
dist_thresh = 100 # UNTESTED
object_lock = False
no_detect_counter = 0

def search_loop(result):
    global no_detect_counter, object_lock
    detected = result['detect']
    dist = result['dist']
    cat = result['class'] # if cat is not 0, this is a list
    if detected is False: 
        if object_lock is False or no_detect_counter > 5: # Either the object detection model hasn't found the object yet or it has but it has been 5 predictions since we had a detection
            moveforward(0.1)
        else: # keep still
            no_detect_counter +=1
        waitToStill()
        return False
    if object_lock is False:
        object_lock = True # Sets this to true the first time the prediction detects an object. Is released once tagging is complete before proceeding to the next doll
    no_detect_counter = 0
    if dist > dist_thresh or dist < -dist_thresh:

        turn_angle = dist*turn_const # if dist is negative, will turn right
        robot.rotate(str(turn_angle))
        waitToStill()
        return False
    if cat == 0: # cat is int
        moveforward(0.1)
        return False
    else:
        if is_full_match(cat): 
            save_target_coords()
            flash_green()
        else:
            flash_red()
        
        tag_count += 1
        object_lock = False
        binary_lock = False
        if tag_count == 3:
            return True
        else: 
            
            align(center_coords)
            if tag_count == 1: robot.rotate('-45')
            else : robot.rotate('45')
            waitToStill()
            return False


def rescue_loop(result):
    global no_detect_counter, object_lock
    detected = result['detect']
    dist = result['dist']
    if detected is False: 
        if no_detect_counter > 5: # Either the object detection model hasn't found the object yet or it has but it has been 5 predictions since we had a detection
            moveforward(0.05)
        else: # keep still
            no_detect_counter +=1
        waitToStill()
        return False
    no_detect_counter=0

    if dist > dist_thresh or dist < -dist_thresh:
        turn_angle = dist*turn_const # if dist is negative, will turn right
        robot.rotate(str(turn_angle))
        waitToStill()
        return False
    robot.closearm()
    if is_gripped(): # need to test
        robot.movearm('x 0 y 50')
        return True
    else:
        robot.openarm()
        robot.move('x 0.1 vxy 0.15') # inch forward. Check for a better way.
        return False

tmp_frame = None # Currently not used anywhere. Can delete later.

def s_and_r(targets): # target is a list
    global tmp_frame
    target_cats = targets
    moveforward(0.4)
    save_center_coords()
    robot.rotate('-135')
    robot.startvideo()
    # Search loop
    while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
        pass

    initial_setup = False
    search_completed = False
    pickup_completed = False
    pickup_setup = False
    while True:
        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        tmp_frame = robot.frame
        cv2.imshow('Live video', tmp_frame) # access the video feed by robot.frame
    
        
        if initial_setup is False:
            initial_setup = set_grip_threshold()
        
        elif search_completed is False:
            if binary_lock is False:
                cropped_frame = crop_frame_by(tmp_frame,7)
                result = detect_binary(cropped_frame) # BINARY CLASSIFIER
                lock_on_loop(result)
            else:
                result = detect_object(tmp_frame) # OBJECT DETECTOR.. 
                search_completed = search_loop(result)
        elif pickup_completed is False:
            if pickup_setup is False:
                rescue_grip()
                if not target_coords:
                    print('No target found during search phase. Picking up closest doll.')
                    align(partials_coords[0])
                    # robot.exit()
                    # break
                else: align(target_coords) # align robot to the coordinates it saved when it detected the correct doll
                pickup_setup = True
            result = detect_object(tmp_frame) # OBJECT DETECTOR
            pickup_completed = rescue_loop(result)            
        else:
            robot.exit()
            break
                
            
        k = cv2.waitKey(16) & 0xFF
        if k == 27: # press esc to stop
            print("Quitting")
            robot.exit()
            break
