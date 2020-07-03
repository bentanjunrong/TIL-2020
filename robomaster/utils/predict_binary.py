import cv2
import numpy as np
import tensorflow as tf
# from EP_api import Robot, findrobotIP
# from frame_processing import *

model = tf.keras.models.load_model("../models/RESNET50_binary_snapshot_01.h5")

def binary_detect(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image).reshape((1, 224, 224, 3))
    res = model.predict(image)   
    if res >= 0.5:
        pred = 1
    else:
        pred = 0
    return pred #set to res for probability


# TEST 
# robot = Robot(findrobotIP())
# robot.startvideo()
# while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
# 	pass


# while True:
#     # cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
#     # cv2.imshow('Live video', robot.frame) # access the video feed by robot.frame
#     frame = crop_frame_by(robot.frame,7)
#     try:
#         # analyse and return res
#         res = detect_binary(frame)
#         print(res)
#     except:
#         pass

#     # Write the frame with the detection boxes
#     cv2.imshow('fk this shit', frame)
#     k = cv2.waitKey(16) & 0xFF
#     if k == 27: # press esc to stop
#         print("Quitting")
#         robot.exit()
#         break
