import cv2
import time
import torch
import imutils
import numpy as np
import matplotlib.pyplot as plt
from yolo.YOLOv3 import YOLOv3Predictor

# use gpu if can
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# modanet params
yolo_modanet_params = {
    "model_def": "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/yolov3-modanet.cfg",
    "weights_path": "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/yolov3-modanet_last.weights", # using modanet pretrained weights
    "class_path":"/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/modanet.names", # using modanet classes
    "conf_thres": 0.1,
    "nms_thres":0.4,
    "img_size": 416,
    "device": device
}

colors = np.array([plt.get_cmap("rainbow")(i) for i in np.linspace(0, 1, 13)])

classes = None
classesFile = "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/modanet.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def detect_object(frame):
    res = { "detect": 0, "dist": None, "class": None }
    mapping = { 8:1, 7:2, 4:3, 5:4, 10:5 }
    
    detections = YOLOv3Predictor(params=yolo_modanet_params).get_detections(frame)
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
        if int(cls_pred) not in [4,5,7,8,10]: continue

        res["detect"] = 1

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        height, width, channels = frame.shape

        centre = (width // 2, height // 2)
        centre_bb = ((x1 + x2) // 2, (y1 + y2) // 2)

        res["dist"] = centre_bb[0] - centre[0]
        res["class"] = mapping[int(cls_pred)]

        print('Dist from centre:', res["dist"], ", Class:", res["class"], "Confidence:", cls_conf)

    return res

cv2.namedWindow('fk this shit', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture('TIL_vidtest.mp4')

while not cap.isOpened():
    cap = cv2.VideoCapture('TIL_vidtest.mp4')
    print("loading video...")
    # esc to close stream
    if cv2.waitKey(16) == 27:
        break

cnt = 0
while cap.isOpened():

    # get frame from the video
    hasFrame, frame = cap.read()

    # q to quit
    if cv2.waitKey(16) == ord("q"):
        break

    print(cnt)
    if cnt < 200 or cnt % 30:
        cnt += 1
        continue
    cnt += 1

    # stop the program if reached end of video
    if not hasFrame: break

    detections = YOLOv3Predictor(params=yolo_modanet_params).get_detections(frame)
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
        if int(cls_pred) not in [4,5,7,8,10]: continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        height, width, channels = frame.shape

        centre = (width // 2, height // 2)
        centre_bb = ((x1 + x2) // 2, (y1 + y2) // 2)

        print('Dist from centre:')
        print(((x1 + x2) / 2) - (width / 2))

        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))       

        color = colors[int(cls_pred)]
        color = tuple(c*255 for c in color)
        color = (.7*color[2],.7*color[1],.7*color[0])       
        font = cv2.FONT_HERSHEY_SIMPLEX   
        text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
        cv2.rectangle(frame,(x1,y1) , (x2,y2) , color,3)
        y1 = 0 if y1<0 else y1
        y1_rect = y1-25
        y1_text = y1-5
        if y1_rect<0:
            y1_rect = y1+27
            y1_text = y1+20
        cv2.rectangle(frame,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)

    # Write the frame with the detection boxes
    cv2.imshow('fk this shit', frame)
