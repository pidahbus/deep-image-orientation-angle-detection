#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
sys.path.append("..")


# In[12]:



# In[13]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[14]:


# Import Libraries
from transformers import TFAutoModel
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


# In[15]:


from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

class ViTRotGenerator(Sequence):
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

        X_conv = []
        X_vit = []
        y = []
        
        for i, f in enumerate(batch_files):
            try:
                angle = float(np.random.choice(range(0, 360)))
                img = rotate_preserve_size(f, angle, (self.dim, self.dim))
                img = np.array(img)
                X_vit.append(img)

                img = np.expand_dims(img, axis=0)
                X_conv.append(img)
                y.append(angle)

            except:
                pass
        
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X_conv = np.concatenate(X_conv, axis=0)
        y = np.array(y)

        return [X_vit, X_conv], y
    
    def on_epoch_end(self):
        random.shuffle(self.files)


# In[16]:


class ViTValidationTestGenerator(Sequence):
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
        

        X_conv = []
        X_vit = []
        y = []
        
        for i in range(len(df_batch)):
            try:
                angle = df_batch.angle[i]
                path = os.path.join(self.image_dir, df_batch.image[i])
                img = rotate_preserve_size(path, angle, (self.dim, self.dim))

                img = np.array(img)
                X_vit.append(img)

                img = np.expand_dims(img, axis=0)
                X_conv.append(img)
                y.append(angle)

            except:
                pass
        
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X_conv = np.concatenate(X_conv, axis=0)
        y = np.array(y)

        return [X_vit, X_conv], y


# In[17]:


# get ViT base model
vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")


# In[18]:


IMAGE_SIZE=224
PATCH_SIZE = 16
PROJECTION_DIM = 768


# In[19]:


# get CONV base model
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
for layer in conv_base.layers:
    layer.trainable = False


# In[20]:


# Define model
vit_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
vit_out = vit_base(vit_input)[1]
vit_out = L.Dense(512, activation="relu")(vit_out)

conv_input = L.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
conv_out = conv_base(conv_input)
conv_out = L.Flatten()(conv_out)
conv_out = L.Dense(512, activation="relu")(conv_out)

x = L.Concatenate()([vit_out, conv_out])
x = L.Dense(512, activation="relu")(x)
x = L.Dense(256, activation="relu")(x)
x = L.Dense(64, activation="relu")(x)
y = L.Dense(1, activation="linear")(x)

model = Model([vit_input, conv_input], y)
model.summary()


# In[21]:


model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.1))

train_gen = ViTRotGenerator("/data/chandanp/train2017/", 8, IMAGE_SIZE)
val_gen = ViTValidationTestGenerator(image_dir="/data/subhadip/data/validation-test/", 
                                     df_label_path="/data/subhadip/data/validation-test.csv",
                                     batch_size=8, dim=IMAGE_SIZE, mode="valid")
cp = ModelCheckpoint("/data/subhadip/weights/model-multi-ang-loss.h5", save_weights_only=False, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
csv_logger = CSVLogger("/data/subhadip/weights/model-multi-ang-loss.csv")
es = EarlyStopping(monitor="val_loss", patience=5)
model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[cp, es, reduce_lr, csv_logger])


# In[18]:


model.evaluate(val_gen, steps=2)


# In[ ]:




