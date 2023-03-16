#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[3]:


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
from tensorflow_addons.optimizers import AdamW


# In[149]:


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


# In[150]:


class ViTValidationTestGenerator(Sequence):
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


# In[4]:


# get ViT base model
vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")
for layer in vit_base.layers:
    layer.trainable = True

# In[72]:


IMAGE_SIZE=224
PATCH_SIZE = 16
PROJECTION_DIM = 768


# In[145]:


# get CONV base model
conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
for layer in conv_base.layers:
    layer.trainable = True


# In[143]:


class PatchAttention(L.Layer):
    def __init__(self, projection_dim):
        super(PatchAttention, self).__init__()
        self.mha = L.MultiHeadAttention(num_heads=1, key_dim=projection_dim)
        
    def call(self, encoded_patches, image_size, patch_size):
        batch_size = tf.shape(encoded_patches)[0]
        max_seq_len = tf.shape(encoded_patches)[1]
        # x = L.LayerNormalization(epsilon=1e-6)(encoded_patches)
        x = encoded_patches
        _, attention_weights = self.mha(x, x, return_attention_scores=True)
        attention_weights = tf.reshape(attention_weights, shape=(batch_size, max_seq_len, max_seq_len))
        attention_weights = tf.math.reduce_mean(attention_weights, axis=1)

        # Removing CLS token
        attention_weights = attention_weights[:,1:]
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        patches = image_size // patch_size
        attention_weights = tf.reshape(attention_weights, shape=(batch_size, patches, patches))

        # move to image space
        pixel_weights = tf.repeat(attention_weights, repeats=[patch_size], axis=-1)
        pixel_weights = tf.repeat(pixel_weights, repeats=[patch_size], axis=1) 
        pixel_weights = tf.expand_dims(pixel_weights, axis=-1)
        
        return pixel_weights



class PatchAttentionV2(L.Layer):
    def __init__(self, projection_dim):
        super(PatchAttentionV2, self).__init__()
        self.mha = L.MultiHeadAttention(num_heads=1, key_dim=projection_dim)
        
    def call(self, encoded_patches, image_size, patch_size):
        batch_size = tf.shape(encoded_patches)[0]
        
        _, attention_weights = self.mha(encoded_patches, encoded_patches, return_attention_scores=True)
        attention_weights = tf.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, 1:, 0]
        attention_weights = tf.nn.sigmoid(attention_weights)
        
        patches = image_size // patch_size
        attention_weights = tf.reshape(attention_weights, shape=(batch_size, patches, patches))

        # move to image space
        pixel_weights = tf.repeat(attention_weights, repeats=[patch_size], axis=-1)
        pixel_weights = tf.repeat(pixel_weights, repeats=[patch_size], axis=1)
        pixel_weights = tf.expand_dims(pixel_weights, axis=-1)
        
        return pixel_weights


# In[148]:


# Define model
vit_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
vit_out = vit_base(vit_input)
pixel_weights = PatchAttentionV2(PROJECTION_DIM)(vit_out[0], IMAGE_SIZE, PATCH_SIZE)

conv_input = L.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = L.Multiply()([pixel_weights, conv_input])
x = conv_base(x)
x = L.Flatten()(x)
x = L.Dense(512, activation="relu")(x)
x = L.Dense(256, activation="relu")(x)
x = L.Dense(64, activation="relu")(x)
y = L.Dense(1, activation="linear")(x)

model = Model([vit_input, conv_input], x)
model.summary()


# In[ ]:


model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.1))
# model.compile(loss=angular_loss_mae, optimizer=AdamW(learning_rate=0.01, weight_decay=0.0001))
model.load_weights("/data/subhadip/weights/model-vit-en-ang-loss.h5")

train_gen = ViTRotGenerator("/data/chandanp/train2017/", 16, IMAGE_SIZE)
val_gen = ViTValidationTestGenerator(image_dir="/data/subhadip/data/", 
                                  df_label_path="/data/subhadip/data/validation-test.csv",
                                  batch_size=16, dim=IMAGE_SIZE, mode="valid")
cp = ModelCheckpoint("/data/subhadip/weights/model-vit-en-ang-loss.h5", save_weights_only=True, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
es = EarlyStopping(monitor="val_loss", patience=5)
model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[reduce_lr, cp, es])

