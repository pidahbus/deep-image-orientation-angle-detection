import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import CategoricalAccuracy
import numpy as np
from sklearn.metrics import accuracy_score

def angular_loss_mae(y_true, y_pred):
    N = tf.math.divide(y_pred, 360)
    N = tf.floor(N)

    y_pred = K.switch(K.greater_equal(y_pred, 360), y_pred - (N * 360), y_pred)
    y_pred = K.switch(K.less(y_pred, 0), y_pred + (K.abs(N) * 360), y_pred)

    error1 = K.abs(y_pred - y_true)
    error2 = 360 - error1
    error = K.minimum(error1, error2)
    mae = K.mean(error)

    return mae

def angular_loss_mae_deprecated(y_true, y_pred):
    error1 = K.abs(y_pred - y_true)
    error2 = 360 - error1
    error = K.minimum(error1, error2)
    mae = K.mean(error)

    return mae


def sinusoidal_loss(y_true, y_pred):
    loss = tf.math.subtract(y_true, y_pred)
    loss = tf.math.divide(loss, 2.0)
    loss = tf.math.sin(loss)
    loss = tf.abs(loss)
    loss = tf.math.reduce_mean(loss)
    loss = 90.0 * loss
    return loss

def circular_loss(y_true, y_pred):
    N = tf.math.divide(y_pred, 360)
    N = tf.floor(N)

    y_pred = K.switch(K.greater_equal(y_pred, 360), y_pred - (N * 360), y_pred)
    y_pred = K.switch(K.less(y_pred, 0), y_pred + (K.abs(N) * 360), y_pred)

    loss = 180**2 - (180 - K.abs(y_true - y_pred))**2
    return K.mean(loss)



def custom_accuracy(y_true, y_pred):
    y_true = y_true.reshape(-1, )
    y_pred = y_pred.reshape(-1, )
    y_pred = np.mod(y_pred, 360)

    y_pred_class = []
    for y in y_pred:
        if y <= 11.25:
            y_pred_class.append(0.0)
        elif y <= 33.75:
            y_pred_class.append(22.5)
        elif y <= 56.25:
            y_pred_class.append(45.0)
        elif y <= 78.75:
            y_pred_class.append(67.5)
        elif y <= 101.25:
            y_pred_class.append(90.0)
        elif y <= 123.75:
            y_pred_class.append(112.5)
        elif y <= 146.25:
            y_pred_class.append(135.0)
        elif y <= 168.75:
            y_pred_class.append(157.5)
        elif y <= 191.25:
            y_pred_class.append(180.0)
        elif y <= 213.75:
            y_pred_class.append(202.5)
        elif y <= 236.25:
            y_pred_class.append(225.0)
        elif y <= 258.75:
            y_pred_class.append(247.5)
        elif y <= 281.25:
            y_pred_class.append(270.0)
        elif y <= 303.5:
            y_pred_class.append(292.5)
        elif y <= 326.25:
            y_pred_class.append(315.0)
        elif y <= 348.75:
            y_pred_class.append(337.5)
        elif y <= 360:
            y_pred_class.append(0.0)

    y_pred_class = np.array(y_pred_class)
    y_true = y_true.astype("str")
    y_pred_class = y_pred_class.astype("str")

    acc = accuracy_score(y_true, y_pred_class)

    return acc


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.float32)])
def tf_custom_accuracy(y_true, y_pred):    
    acc = tf.numpy_function(custom_accuracy, [y_true, y_pred], tf.double)
    return acc