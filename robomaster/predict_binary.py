import cv2
import numpy as np
import tensorflow as tf

# image = cv2.imread("VideoFrame_uncropped2.jpg")


model = tf.keras.models.load_model("C:\\Users\\Raphael\\Documents\\TIL_yolo_model\\Xception_binary_snapshot_01.h5")
res = model.predict(image)
def detect_binary(image):
    image = cv2.resize(image, (1280, 1280))
    image = np.array(image).reshape((1, 1280, 1280, 3))
    res = model.predict(image)   
    return res

# print(predict_binary(image))