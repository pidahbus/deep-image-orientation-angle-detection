#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.append("..")


# In[2]:


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
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from loguru import logger
from tensorflow.keras.utils import Sequence


# In[3]:


# #Define conv base
# conv_base = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
# for layer in conv_base.layers:
#     layer.trainable = False


# In[9]:


# Define model
img_input = L.Input(shape=(40, 40, 3))
x = L.Flatten()(img_input)
x = L.Dense(512, activation="relu")(x)
x = L.BatchNormalization()(x)
x = L.Dense(256, activation="relu")(x)
x = L.BatchNormalization()(x)
x = L.Dense(64, activation="relu")(x)
x = L.BatchNormalization()(x)
y = L.Dense(1, activation="linear")(x)
model = Model(img_input, y)

model.summary()


# In[10]:


# Batch Generator
class RotGenerator(Sequence):
    def __init__(self, image_dir, batch_size, dim):
        self.files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.batch_size = batch_size
        self.dim = dim
        
    def __len__(self):
        if len(self.files) % self.batch_size == 0:
            return len(self.files) // self.batch_size
        return len(self.files) // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_files = self.files[batch_slice]
        
        X = np.zeros(shape=(len(batch_files), self.dim, self.dim, 3))
        y = np.zeros(shape=(len(batch_files), ))
        
        for i, f in enumerate(batch_files):
            angle = float(np.random.choice(range(0, 360)))
            img = rotate_preserve_size(f, angle, (self.dim, self.dim))
            
            X[i] = img
            y[i] = angle
        
        return X, y
    
    def on_epoch_end(self):
        random.shuffle(self.files)
            


# In[17]:


# train
# model = tf.keras.models.load_model("/users/phd/subhadip/CS776A/weights/model-ffnn-ang-loss.h5")
model.compile(loss=angular_loss_mae, optimizer="adam")

train_gen = RotGenerator("/data/subhadip/data/train2017_Resize/", 1024, 40)
cp = ModelCheckpoint("/users/phd/subhadip/CS776A/weights/model-ffnn-ang-loss.h5", save_weights_only=False, 
                     save_best_only=True, monitor="loss")
es = EarlyStopping(monitor="loss", patience=5)
model.fit(train_gen, epochs=10000, callbacks=[cp, es])


# In[ ]:




