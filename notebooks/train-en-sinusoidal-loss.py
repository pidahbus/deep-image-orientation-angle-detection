#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


# Import Libraries
from utils import rotate_preserve_size
from loss import angular_loss_mae, sinusoidal_loss
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


# In[4]:


#Define conv base
# conv_base = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
for layer in conv_base.layers:
    layer.trainable = True


# In[5]:


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

model.summary()


# In[6]:


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
            # img = cv2.imread(f)
            angle = float(np.random.choice(range(0, 360)))
            img = rotate_preserve_size(f, angle, (self.dim, self.dim))
            
            X[i] = img
            y[i] = angle
        
        return X/255., y
    
    def on_epoch_end(self):
        random.shuffle(self.files)
            


# In[7]:


# Batch Generator
class ValidationTestGenerator(Sequence):
    def __init__(self, image_dir, df_label_path, batch_size, dim, mode):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.mode = mode
        
        df_label = pd.read_csv(df_label_path)
        self.df = df_label[df_label["mode"] == self.mode].reset_index(drop=True)
        
    def __len__(self):
        total = self.df.shape[0]
        if total % self.batch_size == 0:
            return total // self.batch_size
        return total // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        df_batch = self.df[batch_slice].reset_index(drop=True).copy()
        
        X = np.zeros(shape=(len(df_batch), self.dim, self.dim, 3))
        y = np.zeros(shape=(len(df_batch), ))
        
        for i in range(len(df_batch)):
            angle = df_batch.angle[i]
            path = os.path.join(self.image_dir, df_batch.image[i])
            img = rotate_preserve_size(path, angle, (self.dim, self.dim))
            
            X[i] = img
            y[i] = angle
        
        return X/255., y
    
    def on_epoch_end(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
            


# In[8]:


# train
model.compile(loss=sinusoidal_loss, optimizer=Adadelta(learning_rate=0.1), metrics=[angular_loss_mae])

train_gen = RotGenerator("/data/chandanp/train2017/", 32, 299)
val_gen = ValidationTestGenerator(image_dir="/data/subhadip/data/validation-test/", 
                                  df_label_path="/data/subhadip/data/validation-test.csv",
                                  batch_size=32, dim=299, mode="valid")
cp = ModelCheckpoint("/data/subhadip/weights/model-en-sin-loss.h5", save_weights_only=False, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
es = EarlyStopping(monitor="val_loss", patience=5)
model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[cp, es, reduce_lr])


# In[ ]:




