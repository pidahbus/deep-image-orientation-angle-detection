#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[64]:


import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from utils import rotate_preserve_size
import cv2
import random

from layers import mlp, Patches, PatchEncoder
from loss import angular_loss_mae
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import Sequence

from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

from layers import EuclideanPOS


# In[78]:


learning_rate = 0.001
weight_decay = 0.0001
IMAGE_SIZE = 224
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (IMAGE_SIZE // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]
input_shape = (224, 224, 3)


# In[79]:


class RotGenerator(Sequence):
    def __init__(self, image_dir, batch_size, dim, channels_first=False, is_vit=False):
        self.files = glob.glob(os.path.join(image_dir, "*.jpg"))
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

        X = []
        y = []
        
        for i, f in enumerate(batch_files):
            try:
                angle = float(np.random.choice(range(0, 360)))
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
            if not self.channels_first:
                X = X.transpose(0, 2, 3, 1)
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
                y.append(angle)

            except:
                pass
        
        if self.is_vit:
            X = feature_extractor(images=X, return_tensors="pt")["pixel_values"]
            X = np.array(X)
            if not self.channels_first:
                X = X.transpose(0, 2, 3, 1)
        else:
            X = np.concatenate(X, axis=0)
        y = np.array(y)

        return X, y


# In[80]:





# In[81]:


def create_vit_model():
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    
    euc_pos_encoded = EuclideanPOS(IMAGE_SIZE)(inputs)
    
    patches = Patches(patch_size)(euc_pos_encoded)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    y = layers.Dense(1, activation="linear")(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=y)
    print(model.summary())
    return model


# In[82]:


model = create_vit_model()


# In[83]:


model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.1))
model.load_weights("/data/subhadip/weights/model-euc-vit-ang-loss.h5")

train_gen = RotGenerator("/data/chandanp/train2017/", 16, IMAGE_SIZE, is_vit=True)
val_gen = ValidationTestGenerator(image_dir="/data/subhadip/validation-test/", 
                                  df_label_path="/data/subhadip/validation-test.csv",
                                  batch_size=32, dim=IMAGE_SIZE, mode="valid", is_vit=True)
cp = ModelCheckpoint("/data/subhadip/weights/model-euc-vit-ang-loss.h5", save_weights_only=True, 
                     save_best_only=True, monitor="loss")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
es = EarlyStopping(monitor="val_loss", patience=5)
model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[cp, es, reduce_lr])


# In[ ]:




