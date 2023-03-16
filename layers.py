import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers as L



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


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


class EuclideanPOS(layers.Layer):
    def __init__(self, image_size):
        super(EuclideanPOS, self).__init__()
        self.w = self.add_weight(shape=(image_size, image_size, 3), trainable=True, initializer="glorot_normal")
        
        cx, cy = ((image_size-1)/2, (image_size-1)/2)
        xcoords, ycoords = tf.meshgrid(np.arange(image_size).astype("float32"), 
                                       np.arange(image_size).astype("float32"))
        self.euc_pos = tf.sqrt((xcoords - cx)**2 + (ycoords - cy)**2) / image_size
        self.euc_pos = tf.expand_dims(self.euc_pos, axis=-1)
        
    def call(self, img_arr):
        return img_arr + self.w + self.euc_pos