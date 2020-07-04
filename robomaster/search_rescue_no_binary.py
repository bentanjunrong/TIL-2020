from utils.EP_api import Robot, findrobotIP
import time
import cv2
import numpy
from utils.object_detector import object_detect
from utils.predict_binary import binary_detect
from utils.nlp_robo import process_text


robot = Robot(findrobotIP())
target_cats = 0
center_coords = []
grip_thresh = 0
boundaries = [0,0.93,-0.75,0.65] # xmin, xmax, ymin, ymax in meters

full_coords = [] # fully correct detections go here
partials_coords = [] # partially correct detections go here
none_coords = [] # detections with no match go here

target_coords = [] # Rescue target coords
red_coords = [] # Compiled list of bad match coords. If target_coords is empty, a coord will be randomly chosen for rescue from here.


# For Lock On
binary_lock = False # when an object is detected, set to True so that if subsequently there are no detections, the robot doesn't try to rotate. Will be reset once tagging is complete


# For Object detection
tag_count = 0
turn_const = 0.06 # 0.02 UNTESTED
dist_thresh = 30 # 45 # UNTESTED
object_lock = False
no_detect_counter = 0
NO_DETECT_CNT_MAX = 10


tmp_frame = None # Currently not used anywhere. Can delete later.


def expandBoundaries(val):
    global boundaries
    boundaries[0] -= val
    boundaries[1] += val
    boundaries[2] -= val
    boundaries[3] += val

def waitToStill(): #tested
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

def outOfBounds(dist): #Tested
    x_cur, y_cur = robot._sendcommand('chassis position ?').split(' ')[:2]
    angle = nearest_90_angle()
    x_new, y_new = x_cur,y_cur
    x_new = float(x_new)
    y_new = float(y_new)
    if angle == 0: x_new += dist
    elif angle == 180: x_new -= dist
    elif angle == -90: y_new -= dist
    else: y_new += dist
    if x_new < boundaries[0] or x_new > boundaries[1] or y_new < boundaries[2] or y_new > boundaries[3]:
        return True
    else: return False


def moveforward(dist): # UNtested
    if(outOfBounds(dist)): # function to check if moving this far will cause the robot to go out of bounds
        return False
    robot.move('x {} vxy 0.1'.format(str(dist)))
    waitToStill()
    return True

def get_current_angle(): #tested
    current_angle = float(robot._sendcommand('chassis position ?').split(' ')[2]) # Relative to power-up/starting position, which should be facing north
    if current_angle > 180: current_angle -= 360
    return current_angle
    # return 90 #TEST

def turn_to_angle(final_angle): #tested
    current_angle = get_current_angle()
    angle_to_turn = final_angle - current_angle
    robot.rotate(str(angle_to_turn))
    waitToStill()

def turn_in_steps(final_angle):
    rotation_step = 5
    current_angle = get_current_angle()
    angle_to_turn = final_angle - current_angle
    if angle_to_turn < 0: rotation_step *= -1
    turn_divisions = angle_to_turn // 5

    for i in range(turn_divisions):
        robot.rotate(str(rotation_step))
        waitToStill()
        time.sleep(1.5)
    turn_to_angle(final_angle)


def nearest_90_angle(): # tested
    angle = get_current_angle()
    if -135 < angle < -45: return -90
    elif -45 <= angle <= 45: return 0
    elif 45 < angle < 135: return 90
    else: return 180



# Aligns robot so that it is in the center of the current_loc

# coords is an float list sized 3 containing x,y,z coords of the destination
def align(coords):  # Tested
    x_dest, y_dest, z_dest = coords
    turn_to_angle(z_dest)
    snap_angle = nearest_90_angle() # finds nearest 90 degree angle. I call it snapping lmao
    turn_to_angle(snap_angle)
    x_robot, y_robot = robot._sendcommand('chassis position ?').split(' ')[:2]
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
def crop_frame_by(frame,crop_by_factor,crop_bot=False): #tested
    crop_by = crop_by_factor #lol pls forgive me for this
    width = (1280/crop_by)/2
    x_left = int((640) - width) # from center
    x_right = int((640) + width) #from center
    if crop_bot: return frame[0:625,x_left:x_right]
    else: return frame[0:720,x_left:x_right]



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
    robot.closearm()
    time.sleep(3)
    grip_thresh = calc_image()
    time.sleep(1)
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
    if val > (grip_thresh + 5): return True
    else: return False
def save_current_coords():
    coords = robot._sendcommand('chassis position ?').split(' ')[:3]
    coords = map(float,coords)
    return coords   
    

def save_target_coords(): # Untested
    global target_coords
    target_coords = save_current_coords()

def save_center_coords():# Untested
    global center_coords
    center_coords = save_current_coords()

def flash_green():# Tested
    robot._sendcommand('led control comp all r 0 g 255 b 0 effect blink')
    time.sleep(5)
    robot._sendcommand('led control comp all r 255 g 255 b 255 effect solid')

def flash_red():# Tested
    robot._sendcommand('led control comp all r 255 g 0 b 0 effect blink')
    time.sleep(5)
    robot._sendcommand('led control comp all r 255 g 255 b 255 effect solid')




def lock_on_loop(result): # Untested
    if result: # doll detected
        binary_lock = True
    else: 
        robot.rotate('5') # assuming left-to-right sweep
        waitToStill()



def check_match(cats): # Untested
    matches = []
    for target_cat in target_cats:
        if target_cat in cats:
            matches.append(target_cat)
    if len(matches) == len(target_cats):
        full_coords.append(save_current_coords())
        return 2
    
    elif len(matches): 
        partials_coords.append(save_center_coords())
        return 1
    else: 
        none_coords.append(save_center_coords())
        return 0
        
def tag_dolls():
    global red_coords
    if full_coords:
        target_coords = full_coords.pop()
    else:
        if partials_coords:
            target_coords = partials_coords.pop()
    all_coords = [full_coords,partials_coords,none_coords]
    for i in all_coords:
        if not i:
            continue
        else:
            for j in i:
                red_coords.append(j)
    for coords in red_coords:
        align(coords)
        flash_red()
    if target_coords:
        align(target_coords)
        flash_green()




def search_loop(result): # Untested
    global no_detect_counter, object_lock
    detected = result['detect']
    dist = result['dist']
    cat = result['class'] # if cat is not 0, this is a list
    if detected is False: 
        if object_lock is False or no_detect_counter > NO_DETECT_CNT_MAX: # Either the object detection model hasn't found the object yet or it has but it has been 5 predictions since we had a detection
            moveforward(0.1)
        else: # keep still
            no_detect_counter +=1
        waitToStill()
        return False
    if object_lock is False:
        object_lock = True # Sets this to true the first time the prediction detects an object. Is released once tagging is complete before proceeding to the next doll
    no_detect_counter = 0
    if dist > dist_thresh or dist < -dist_thresh:

        turn_angle = int(dist*turn_const) # if dist is negative, will turn right
        if turn_angle < 5 and turn_angle > 0:
            turn_angle = 5
        elif turn_angle > -5 and turn_angle < 0:
            turn_angle = -5
        robot.rotate(str(turn_angle))
        waitToStill()
        return False
    if cat == 0: # cat is int
        moveforward(0.1)
        return False
    else:
        check_match(cat)
        tag_count += 1
        object_lock = False
        binary_lock = False
        if tag_count == 3:
            tag_dolls()
            return True
        else: 
            
            align(center_coords)
            if tag_count == 1: 
                turn_to_angle(-45)
                turn_in_steps(0)
            else : 
                turn_to_angle(45)
                turn_in_steps(90)
            waitToStill()
            moveforward(0.3)
            return False


def rescue_loop(result): # Untested
    global no_detect_counter, object_lock
    detected = result['detect']
    dist = result['dist']
    if detected is False: 
        if no_detect_counter > NO_DETECT_CNT_MAX: # Either the object detection model hasn't found the object yet or it has but it has been 5 predictions since we had a detection
            moveforward(0.05)
        else: # keep still
            no_detect_counter +=1
        waitToStill()
        return False
    no_detect_counter=0

    if dist > dist_thresh or dist < -dist_thresh:
        turn_angle = dist*turn_const # if dist is negative, will turn right
        if turn_angle < 5 and turn_angle > 0:
            turn_angle = 5
        elif turn_angle > -5 and turn_angle < 0:
            turn_angle = -5
        robot.rotate(str(turn_angle))
        waitToStill()
        return False
    robot.closearm()
    time.sleep(3)
    if is_gripped(): # need to test
        robot.movearm('x 0 y 50')
        flash_green()
        return True
    else:
        robot.openarm()
        time.sleep(3)
        robot.move('x 0.04 vxy 0.15') # inch forward. Check for a better way.
        waitToStill()
        return False


search_completed = False
pickup_completed = False

def s_and_r(targets): # target is a list
    global tmp_frame, target_cats, search_completed,pickup_completed
    robot._sendcommand('led control comp all r 255 g 255 b 255 effect solid')
    target_cats = targets
    moveforward(0.4)
    save_center_coords()
    
    robot.rotate('-135') 
    turn_in_steps(-90)
    moveforward(0.3)
    robot.startvideo()
    # Search loop
    while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
        pass

    initial_setup = False
    
    pickup_setup = False
    while True:
        cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
        tmp_frame = robot.frame
        cv2.imshow('Live video', tmp_frame) # access the video feed by robot.frame
    
        
        if initial_setup is False:
            initial_setup = set_grip_threshold()
        
        elif search_completed is False:
            # if binary_lock is False:
            #     cropped_frame = crop_frame_by(tmp_frame,7)
            #     result = binary_detect(cropped_frame) # BINARY CLASSIFIER
            #     lock_on_loop(result)
            # else:
            cropped_frame = crop_frame_by(tmp_frame,2,crop_bot=True)
            result = object_detect(cropped_frame) # OBJECT DETECTOR.. 
            search_completed = search_loop(result)
        elif pickup_completed is False:
            if pickup_setup is False:
                rescue_grip()
                expandBoundaries(0.2)
                if not target_coords:
                    print('No target found during search phase. Picking up closest doll.')
                    align(red_coords[2])
                    # robot.exit()
                    # break
                else: align(target_coords) # align robot to the target doll coordinates
                pickup_setup = True
            cropped_frame = crop_frame_by(tmp_frame,3,crop_bot=True)
            result = object_detect(cropped_frame) # OBJECT DETECTOR
            pickup_completed = rescue_loop(result)            
        else:
            robot.exit()
            break
                
            
        k = cv2.waitKey(16) & 0xFF
        if k == 27: # press esc to stop
            print("Quitting")
            robot.exit()
            break
# text = ""
# s_and_r(process_text(text))


# TEST ONLY FUNCTIONS REMOVE WHEN COMP STARTS
def start_video():
    robot.startvideo()
    # Search loop
    while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
        pass
def test_claw_algo():
    
    robot.closearm()
    time.sleep(3)
    if is_gripped(): # need to test
        robot.movearm('x 0 y 50')
        flash_green()
        return True
    else:
        robot.openarm()
        time.sleep(3)
        robot.move('x 0.04 vxy 0.1') # inch forward. Check for a better way.
        waitToStill()
        return False
def test_object_centering():
    while True:
        tmp_frame = robot.frame
        if tmp_frame is None: continue
        cropped_frame = crop_frame_by(tmp_frame,2,crop_bot=True)
        result = object_detect(cropped_frame)
        try:
            dist = float(result['dist'])
            print('DIST: ', dist)
            if dist > dist_thresh or dist < -dist_thresh:

                turn_angle = int(dist*turn_const) # if dist is negative, will turn right
                print('ROBOT WILL TURN {} DEGREES'.format(turn_angle))
                robot.rotate(str(turn_angle))
                waitToStill()
            else: break
        except: pass

def binary_classifier_test():
    while True:
        tmp_frame = robot.frame
        if tmp_frame is None: continue
        cropped_frame = crop_frame_by(tmp_frame,7)
        cv2.imshow('binary test',cropped_frame)
        result = binary_detect(cropped_frame)

        print(result)

