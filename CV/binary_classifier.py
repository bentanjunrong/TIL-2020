# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np


# %%
from tensorflow.keras.applications.xception import preprocess_input
train_path = '../../data/binary_classifier_train' # 2 levels up cos this notebook is in a subfolder

train_image_generator = ImageDataGenerator(validation_split = 0.1, preprocessing_function=preprocess_input) 
# Brightness range based on this https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html#PIL.ImageEnhance.Brightness
# horizontal_flip=True,brightness_range=(0.75,1.25), channel_shift_range=100, 


train_data_gen = train_image_generator.flow_from_directory(train_path, subset='training',class_mode='binary')
val_data_gen = train_image_generator.flow_from_directory(train_path, subset='validation',class_mode='binary')


# %%
from tensorflow.keras.applications import Xception
from tensorflow.keras.metrics import TopKCategoricalAccuracy
# Create Model here
model = Sequential()
model.add(Xception(include_top=False, weights='imagenet', pooling='avg'))
model.add(Dense(1,activation='sigmoid'))
model.layers[0].trainable = False

metric = 'accuracy'
model.compile(optimizer='adam', metrics=metric, loss=['binary_crossentropy'])


# %%
from PIL import Image
model.summary()


# %%
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(os.path.join('saved_models','Xception_binary_snapshot_{epoch:02d}.h5'))


# %%
history = model.fit(
            train_data_gen,
            steps_per_epoch=124,
            epochs=5,
            callbacks=checkpoint,
            validation_data=val_data_gen,
            validation_steps=14) # steps tbc
model.save('saved_models/Xception_binary_try2.h5',save_format='h5')

