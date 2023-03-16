#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[16]:


# Import Libraries
from utils import rotate_preserve_size
from loss import angular_loss_mae
import glob
import os
import numpy as np
import cv2
import random

from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.applications import Xception, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from loguru import logger
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adadelta
from generator import RotGenerator, ValidationTestGenerator


# In[12]:


#Define conv base
# conv_base = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
for layer in conv_base.layers:
    layer.trainable = True


# In[13]:


# Define model
img_input = L.Input(shape=(299, 299, 3))
x = conv_base(img_input)
x = L.Flatten()(x)
x = L.Dense(512, activation="relu")(x)
x = L.BatchNormalization()(x)
x = L.Dense(256, activation="relu")(x)
x = L.BatchNormalization()(x)
x = L.Dense(64, activation="relu")(x)
x = L.BatchNormalization()(x)
y = L.Dense(1, activation="linear")(x)
model = Model(img_input, y)

print(model.summary())

# train
model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.1))

train_gen = RotGenerator("/data/chandanp/train2017/", 32, 299)
val_gen = ValidationTestGenerator(image_dir="/data/subhadip/data/validation-test/", 
                                  df_label_path="/data/subhadip/data/validation-test.csv",
                                  batch_size=32, dim=299, mode="valid")
cp = ModelCheckpoint("/data/subhadip/weights/model-en-ang-loss.h5", save_weights_only=False, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
es = EarlyStopping(monitor="val_loss", patience=5)
model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[cp, es, reduce_lr])


# In[ ]:




