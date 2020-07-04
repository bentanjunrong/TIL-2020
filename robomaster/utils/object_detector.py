import cv2
import colorsys
import numpy as np
from yolov4.tf import YOLOv4 # Need to change detection threshold in utilities/predict
# from EP_api import Robot, findrobotIP # COMMENT OUT WHEN DONE TESTING


DETECT_THRES = 0.1
HEIGHT_THRES = 0.3 # BBox must fill up at least 40% of the screen
# load yolo model
yolo = YOLOv4()
yolo.classes = "../models/custom_data/custom.names"
yolo.make_model()
yolo.load_weights("../models/yolov4-custom_best.weights", weights_type="yolo")

def object_detect(frame):
    res = { "detect": 0, "dist": None, "class": 0 }

    # analyse frame
    try:
        bboxes = yolo.predict(frame)
    except:
        bboxes = []
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
    # initialize frame values
    height, width, channels = frame.shape
    centre = (width // 2, height // 2)
    
    # process bbs
    seen = []
    y_range = [9999999,9999999] # y_min of the highest bbox and y_max of the lowest bbox

    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        bbox_height = int(bbox[3])
    
        y_min = c_y - bbox_height
        y_max = c_y + bbox_height
        if y_min < y_range[0]: y_range[0] = y_min
        if y_max < y_range[1] : y_range[1] = y_max

        class_id = int(bbox[4]) + 1 # + 1 because original range is 0 to 4
        cls_pred = bbox[5]
        if cls_pred < DETECT_THRES or int(class_id) in seen: continue
        seen.append(int(class_id))
        centre_bb = (c_x, c_y)
        
        res["dist"].append(centre_bb[0] - centre[0])
        res["class"].append(class_id)

        print('Dist from centre:', centre_bb[0] - centre[0], ", Class:", class_id, "Height:", y_min, y_max)

    res["dist"] = sum(res["dist"]) / len(res["dist"])
    if HEIGHT_THRES > (y_range[1] - y_range[0])/height: res["class"] = 0
    
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
def crop_frame_by(frame,crop_by_factor):
    crop_by = crop_by_factor #lol pls forgive me for this
    width = (1280/crop_by)/2
    x_left = int((640) - width) # from center
    x_right = int((640) + width) #from center
    return frame[0:625,x_left:x_right]

# ## TEST
# robot = Robot(findrobotIP())
# robot.startvideo()
# # robot._sendcommand('robotic_arm moveto x 182 y 0')
# while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
# 	pass


# while True:
#     # cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
#     # cv2.imshow('Live video', robot.frame) # access the video feed by robot.frame
#     frame = crop_frame_by(robot.frame,2)
#     try:
#         # analyse and return res
#         res = object_detect(frame)
#         print(res)

#         # analyse and draw - comment out to improve performance
#         bboxes = yolo.predict(frame)
#         frame = draw_bbox(frame, bboxes, yolo.classes)
#     except:
#         pass

#     # Write the frame with the detection boxes
#     cv2.imshow('fk this shit', frame)
#     k = cv2.waitKey(16) & 0xFF
#     if k == 27: # press esc to stop
#         print("Quitting")
#         robot.exit()
#         break

#### Commented all these out cos they will get triggered during import in EP_s_and_r.py
# cv2.namedWindow('fk this shit', cv2.WINDOW_NORMAL)
# cap = cv2.VideoCapture('../TIL_vidtest.mp4')

# while not cap.isOpened():
#     cap = cv2.VideoCapture('../TIL_vidtest.mp4')
#     print("loading video...")
#     # esc to close stream
#     if cv2.waitKey(16) == 27:
#         break

# cnt = 0
# while cap.isOpened():

#     # get frame from the video
#     hasFrame, frame = cap.read()

#     # q to quit
#     if cv2.waitKey(16) == ord("q"):
#         break

#     print(cnt)
#     cnt += 1
#     if (cnt - 1) < 400 or (cnt - 1) % 30: continue

#     # stop the program if reached end of video
#     if not hasFrame: break

#     try:
#         # analyse and return res
#         res = detect_object(frame)
#         print(res)

#         # analyse and draw - comment out to improve performance
#         bboxes = yolo.predict(frame)
#         frame = draw_bbox(frame, bboxes, yolo.classes)
#     except:
#         pass

#     # Write the frame with the detection boxes
#     cv2.imshow('fk this shit', frame)
