import cv2
import colorsys
import numpy as np

DETECT_THRES = 0.2

# need to edit /utility/predict.py
from yolov4.tf import YOLOv4
## line 140 change to: return np.concatenate(bboxes, axis=0) if bboxes else []
## line 207 add: if not len(bboxes): return []

# load yolo model
yolo = YOLOv4()
yolo.classes = "../../s3_model/custom_data/custom.names"
yolo.make_model()
yolo.load_weights("../../s3_model/yolov4-custom_best.weights", weights_type="yolo")

def detect_object(frame):
    res = { "detect": 0, "dist": None, "class": 0 }

    # analyse frame
    bboxes = yolo.predict(frame)
    if len(bboxes): print(max([x[5] for x in bboxes]))
    if not len(bboxes) or max([x[5] for x in bboxes]) < DETECT_THRES: return res

    # unnormalize image points
    image = np.copy(frame)
    height, width, _ = image.shape
    max_size = max(height, width)
    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    # initialise res vals
    res["detect"] = 1
    res["dist"] = []
    res["class"] = []

    # process bbs
    seen = []
    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        class_id = int(bbox[4])
        cls_pred = bbox[5]
        if cls_pred < DETECT_THRES or int(class_id) in seen: continue
        seen.append(int(class_id))

        height, width, channels = frame.shape
        centre = (width // 2, height // 2)
        centre_bb = (c_x, c_y)

        res["dist"].append(centre_bb[0] - centre[0])
        res["class"].append(class_id)

        print('Dist from centre:', centre_bb[0] - centre[0], ", Class:", class_id, "Confidence:", cls_pred)

    res["dist"] = sum(res["dist"]) / len(res["dist"])

    return res

def draw_bbox(image: np.ndarray, bboxes: np.ndarray, classes: dict):
    image = np.copy(image)
    height, width, _ = image.shape
    max_size = max(height, width)

    # Create colors
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(
            lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors,
        )
    )
    
    if not len(bboxes): return image
    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        c_min = (c_x - half_w, c_y - half_h)
        c_max = (c_x + half_w, c_y + half_h)
        class_id = int(bbox[4])
        bbox_color = colors[class_id]
        font_size = min(max_size / 1500, 0.7)
        font_thickness = 1 if max_size < 1000 else 2

        cv2.rectangle(image, c_min, c_max, bbox_color, 3)

        bbox_text = "{}: {:.1%}".format(classes[class_id], bbox[5])
        t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        cv2.rectangle(
            image,
            c_min,
            (c_min[0] + t_size[0], c_min[1] - t_size[1] - 3),
            bbox_color,
            -1,
        )
        cv2.putText(
            image,
            bbox_text,
            (c_min[0], c_min[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image

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
    cnt += 1
    if (cnt - 1) < 400 or (cnt - 1) % 30: continue

    # stop the program if reached end of video
    if not hasFrame: break

    # analyse and return res
    res = detect_object(frame)
    print(res)

    # analyse and draw - comment out to improve performance
    bboxes = yolo.predict(frame)
    frame = draw_bbox(frame, bboxes, yolo.classes)

    # Write the frame with the detection boxes
    cv2.imshow('fk this shit', frame)

#####################################3

# configPath = "../../s3_model/custom_data/cfg_custom/yolov4-custom.cfg"
# weightPath = "../../s3_model/yolov4-custom_best.weights"
# namePath = "../../s3_model/custom_data/custom.names"

# import cv2 as cv

# net = cv.dnn_DetectionModel(configPath, weightPath)
# net.setInputSize(608, 608)
# net.setInputScale(1.0 / 255)
# net.setInputSwapRB(True)

# frame = cv.imread('../testing/99.jpg')

# with open(namePath, 'rt') as f:
#     names = f.read().rstrip('\n').split('\n')

# classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
# for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
#     label = '%.2f' % confidence
#     label = '%s: %s' % (names[classId], label)
#     labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     left, top, width, height = box
#     top = max(top, labelSize[1])
#     cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
#     cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
#     cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# cv.imshow('out', frame)

###################################3

# from ctypes import *
# import math
# import random
# import os
# import cv2
# import numpy as np
# import time
# import darknet


# def convertBack(x, y, w, h):
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax


# def cvDrawBoxes(detections, img):
#     # Colored labels dictionary
#     color_dict = {
#         'tops' : [0, 255, 255],
#         'trousers': [238, 123, 158],
#         'outerwear' : [24, 245, 217],
#         'dresses' : [224, 119, 227],
#         'skirts' : [154, 52, 104]
#     }

#     for detection in detections:
#         x, y, w, h = detection[2][0],\
#             detection[2][1],\
#             detection[2][2],\
#             detection[2][3]
#         name_tag = str(detection[0].decode())
#         for name_key, color_val in color_dict.items():
#             if name_key == name_tag:
#                 color = color_val 
#                 xmin, ymin, xmax, ymax = convertBack(
#                 float(x), float(y), float(w), float(h))
#                 pt1 = (xmin, ymin)
#                 pt2 = (xmax, ymax)
#                 cv2.rectangle(img, pt1, pt2, color, 1)
#                 cv2.putText(img,
#                             detection[0].decode() +
#                             " [" + str(round(detection[1] * 100, 2)) + "]",
#                             (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             color, 2)
#     return img


# netMain = None
# metaMain = None
# altNames = None


# def YOLO():
   
#     global metaMain, netMain, altNames
#     configPath = "../../s3_model/custom_data/cfg_custom/yolov4-custom.cfg"                              # Path to cfg
#     weightPath = "../../s3_model/yolov4-custom_best.weights"                              # Path to weights
#     metaPath = "../../s3_model/custom_data/detector.data"                                 # Path to meta data
#     if not os.path.exists(configPath):                           # Checks whether file exists otherwise return ValueError
#         raise ValueError("Invalid config path `" +
#                          os.path.abspath(configPath)+"`")
#     if not os.path.exists(weightPath):
#         raise ValueError("Invalid weight path `" +
#                          os.path.abspath(weightPath)+"`")
#     if not os.path.exists(metaPath):
#         raise ValueError("Invalid data file path `" +
#                          os.path.abspath(metaPath)+"`")
#     if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
#         netMain = darknet.load_net_custom(configPath.encode( 
#             "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
#     if metaMain is None:
#         metaMain = darknet.load_meta(metaPath.encode("ascii"))
#     if altNames is None:
#         try:
#             with open(metaPath) as metaFH:
#                 metaContents = metaFH.read()
#                 import re
#                 match = re.search("names *= *(.*)$", metaContents,
#                                   re.IGNORECASE | re.MULTILINE)
#                 if match:
#                     result = match.group(1)
#                 else:
#                     result = None
#                 try:
#                     if os.path.exists(result):
#                         with open(result) as namesFH:
#                             namesList = namesFH.read().strip().split("\n")
#                             altNames = [x.strip() for x in namesList]
#                 except TypeError:
#                     pass
#         except Exception:
#             pass
#     cap = cv2.VideoCapture("TIL_vidtest.mp4")                             # Local Stored video detection - Set input video
#     frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
#     frame_height = int(cap.get(4))
#     # Set out for video writer
#     out = cv2.VideoWriter(                                          # Set the Output path for video writer
#         "./output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
#         (frame_width, frame_height))

#     print("Starting the YOLO loop...")

#     # Create an image we reuse for each detect
#     darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
#     while True:                                                      # Load the input frame and write output frame.
#         prev_time = time.time()
#         ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
#         # For Assertion Failed Error in OpenCV
#         if not ret:                                                  # Check if frame present otherwise he break the while loop
#             break

#         frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
#         frame_resized = cv2.resize(frame_rgb,
#                                    (frame_width, frame_height),
#                                    interpolation=cv2.INTER_LINEAR)

#         darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

#         detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
#         image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print(1/(time.time()-prev_time))
#         cv2.imshow('Demo', image)                                    # Display Image window
#         cv2.waitKey(3)
#         out.write(image)                                             # Write that frame into output video
#     cap.release()                                                    # For releasing cap and out. 
#     out.release()
#     print(":::Video Write Completed")

# if __name__ == "__main__":  
#     YOLO()                                                           # Calls the main function YOLO()

####################################3


# use gpu if can
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.empty_cache()

# modanet params
# yolo_modanet_params = {
#     "model_def": "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/yolov3-modanet.cfg",
#     "weights_path": "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/yolov3-modanet_last.weights", # using modanet pretrained weights
#     "class_path":"/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/modanet.names", # using modanet classes
#     "conf_thres": 0.1,
#     "nms_thres":0.4,
#     "img_size": 416,
#     "device": device
# }

# colors = np.array([plt.get_cmap("rainbow")(i) for i in np.linspace(0, 1, 13)])

# classes = None
# classesFile = "/home/rohit/Documents/Computer Science/Hackathons/DSTA TIL/modanet.names"
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')

# def detect_object(frame):
#     res = { "detect": 0, "dist": None, "class": 0 }
#     mapping = { 8:1, 7:2, 4:3, 5:4, 10:5 }
    
#     detections = YOLOv3Predictor(params=yolo_modanet_params).get_detections(frame) # change model to darknet one, use tut online

#     if not detections: return res
#     detections.sort(key=lambda x:x[4], reverse=True)
#     if detections[0][4] < DETECT_THRES: return res

#     res["detect"] = 1
#     res["dist"] = []
#     res["class"] = []

#     seen = []
#     for x1, y1, x2, y2, cls_conf, cls_pred in detections:
#         if int(cls_pred) not in [4,5,7,8,10]: continue # can remove later
#         if cls_pred < DETECT_THRES or int(cls_pred) in seen: continue
#         seen.append(int(cls_pred))

#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         height, width, channels = frame.shape

#         centre = (width // 2, height // 2)
#         centre_bb = ((x1 + x2) // 2, (y1 + y2) // 2)

#         res["dist"].append(centre_bb[0] - centre[0])
#         res["class"].append(mapping[int(cls_pred)])

#         print('Dist from centre:', centre_bb[0] - centre[0], ", Class:", mapping[int(cls_pred)], "Confidence:", cls_conf)

#     res["dist"] = sum(res["dist"]) / len(res["dist"])

#     return res


# detections = YOLOv3Predictor(params=yolo_modanet_params).get_detections(frame)
# for x1, y1, x2, y2, cls_conf, cls_pred in detections:
#     if int(cls_pred) not in [4,5,7,8,10]: continue

#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     height, width, channels = frame.shape

#     centre = (width // 2, height // 2)
#     centre_bb = ((x1 + x2) // 2, (y1 + y2) // 2)

#     print('Dist from centre:')
#     print(((x1 + x2) / 2) - (width / 2))

#     print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))       

#     color = colors[int(cls_pred)]
#     color = tuple(c*255 for c in color)
#     color = (.7*color[2],.7*color[1],.7*color[0])       
#     font = cv2.FONT_HERSHEY_SIMPLEX   
#     text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
#     cv2.rectangle(frame,(x1,y1) , (x2,y2) , color,3)
#     y1 = 0 if y1<0 else y1
#     y1_rect = y1-25
#     y1_text = y1-5
#     if y1_rect<0:
#         y1_rect = y1+27
#         y1_text = y1+20
#     cv2.rectangle(frame,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)