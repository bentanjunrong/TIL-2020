import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../models/RESNET50_binary_snapshot_01.h5")

def predict_binary(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image).reshape((1, 224, 224, 3))
    res = model.predict(image)   
    if res >= 0.5:
        pred = 1
    else:
        pred = 0
    return pred #set to res for probability
