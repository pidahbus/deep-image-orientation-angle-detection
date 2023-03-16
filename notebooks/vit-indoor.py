#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[23]:


# Import Libraries
from transformers import TFAutoModel, ViTForImageClassification
from utils import rotate_preserve_size
from loss import angular_loss_mae, tf_custom_accuracy
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
from metric import Accuracy


# In[4]:


# Parameters
IMAGE_SIZE = 224


# In[61]:


# get vit model
vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")
# vit_base = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")


# In[62]:


# Define model
img_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
x = vit_base(img_input)
y = L.Dense(1, activation="linear")(x[-1])

model = Model(img_input, y)
model.summary()


# In[74]:


from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class RotGenerator(Sequence):
    def __init__(self, image_dir, batch_size, dim, channels_first=False, is_vit=False):
        self.files = []
        for d_path in glob.glob(os.path.join(image_dir, "*")):
            self.files += glob.glob(os.path.join(d_path, "*.jpg"))
        # self.files = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.batch_size = batch_size
        self.dim = dim
        self.channels_first = channels_first
        self.is_vit = is_vit
        
    def __len__(self):
        if len(self.files) % self.batch_size == 0:
            return len(self.files) // self.batch_size
        return len(self.files) // self.batch_size + 1
    
    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_files = self.files[batch_slice]
        
        # X = np.zeros(shape=(len(batch_files), self.dim, self.dim, 3))
        # y = np.zeros(shape=(len(batch_files), ))

        X = []
        y = []
        
        for i, f in enumerate(batch_files):
            try:
                # angle = float(np.random.choice(range(0, 360)))
                angle = float(np.random.choice([0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 
                225.0, 247.5, 270.0, 292.5, 315.0, 337.5, 360.0]))
                img = rotate_preserve_size(f, angle, (self.dim, self.dim))
                img = np.array(img)
                if self.is_vit:
                    X.append(img)
                else:
                    if self.channels_first:
                        img = img.transpose(2, 0, 1)

                    img = np.expand_dims(img, axis=0)
                    X.append(img)
                    # X[i] = img
                    # y[i] = angle
                y.append(angle)

            except:
                pass
        
        if self.is_vit:
            X = feature_extractor(images=X, return_tensors="pt")["pixel_values"]
            X = np.array(X)
        else:
            X = np.concatenate(X, axis=0)
        y = np.array(y)

        return X, y
    
    def on_epoch_end(self):
        random.shuffle(self.files)


# In[83]:


class ValidationTestGenerator(Sequence):
    def __init__(self, image_dir, df_label_path, batch_size, dim, mode, channels_first=False, is_vit=False):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.mode = mode
        self.channels_first = channels_first
        self.is_vit = is_vit
        
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
        
        # X = np.zeros(shape=(len(df_batch), self.dim, self.dim, 3))
        # y = np.zeros(shape=(len(df_batch), ))

        X = []
        y = []
        
        for i in range(len(df_batch)):
            try:
                angle = df_batch.angle[i]
                path = os.path.join(self.image_dir, df_batch.image[i])
                img = rotate_preserve_size(path, angle, (self.dim, self.dim))

                img = np.array(img)
                if self.is_vit:
                    X.append(img)
                else:
                    if self.channels_first:
                        img = img.transpose(2, 0, 1)

                    img = np.expand_dims(img, axis=0)
                    X.append(img)
                    # X[i] = img
                    # y[i] = angle
                y.append(angle)

            except:
                pass
        
        if self.is_vit:
            X = feature_extractor(images=X, return_tensors="pt")["pixel_values"]
            X = np.array(X)
        else:
            X = np.concatenate(X, axis=0)
        y = np.array(y)

        return X, y


# In[85]:



# In[88]:


# train
model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.0001), metrics=[tf_custom_accuracy])
model.load_weights("/data/subhadip/weights/model-vit-ang-loss-indoor.h5")
# acc = Accuracy(batch_size=16)
train_gen = RotGenerator("/data/subhadip/MIT-indoor/Images/", 16, IMAGE_SIZE, is_vit=True)
# val_gen = ValidationTestGenerator(image_dir="/data/subhadip/data/", 
#                                   df_label_path="/data/subhadip/data/validation-test.csv",
#                                   batch_size=32, dim=IMAGE_SIZE, mode="valid", is_vit=True)
cp = ModelCheckpoint("/data/subhadip/weights/model-vit-ang-loss-indoor.h5", save_weights_only=False, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-5)
es = EarlyStopping(monitor="loss", patience=5)
model.fit(train_gen, epochs=10000, callbacks=[cp, es, reduce_lr])


# In[ ]:




