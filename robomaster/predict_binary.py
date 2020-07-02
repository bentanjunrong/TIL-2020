import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("C:\\Users\\Raphael\\Documents\\TIL_yolo_model\\RESNET50_binary_snapshot_01.h5")

def predict_binary(image):
        image = cv2.resize(image, (299, 299))
        image = np.array(image).reshape((1, 299, 299, 3))
        res = model.predict(image)   
        if res >= 0.5:
            pred = 1
        else:
            pred = 0
        return res, pred

# image_list = ["VideoFrame_uncropped23.jpg","VideoFrame_uncropped2.jpg","VideoFrame_uncropped261.jpg","VideoFrame_uncropped284.jpg","VideoFrame_uncropped559.jpg","VideoFrame_uncropped580.jpg","VideoFrame_uncropped1269.jpg","VideoFrame_uncropped1345.jpg"]

# for images in image_list:
#     image = cv2.imread(images)
#     #image = cv2.imread("VideoFrame_uncropped580.jpg")


#     #model = tf.keras.models.load_model("C:\\Users\\Raphael\\Documents\\TIL_yolo_model\\Xception_binary_snapshot_01.h5")
#     #model = tf.keras.models.load_model("C:\\Users\\Raphael\\Documents\\TIL_yolo_model\\RESNET50_binary_snapshot_01.h5")

#     #res = model.predict(image)
    

#     print(predict_binary(image))