# import the necessary packages
#from imutils.video import VideoStream, FileVideoStream
#from imutils.video import FPS
import numpy as np
#import argparse
import imutils
import time
import cv2 

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

path = "C:\\Users\\Raphael\\Documents\\TIL_yolo_model\\"
classesFile = path+"modanet.names"
CLASSES = None
with open(classesFile, 'rt') as f:
    CLASSES = f.read().rstrip('\n').split('\n')

 
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = path+"yolov3-modanet.cfg"
modelWeights = path+"yolov3-modanet_last.weights"


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture('TIL_vidtest.mp4')
#vs = FileVideoStream("TIL_vidtest.mp4")
#time.sleep(2.0)
#fps = FPS().start()
#cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)#added to get it to show
#cv2.waitKey(10)
#print(1)
# loop over the frames from the video stream
while not vs.isOpened():
    vs = cv2.VideoCapture('TIL_vidtest.mp4')
    print("loading video...")

    # esc to close stream
    if cv2.waitKey(16) == 27:
        break

while vs.isOpened():
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    flag, frame = vs.read()
    print("1")
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()	

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > .20:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(10)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    #fps.update()

    # stop the timer and display FPS information
#fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
cv2.waitKey(10)
#vs.stop()
vs.release()