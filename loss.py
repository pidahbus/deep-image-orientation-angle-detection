import tensorflow as tf
import tensorflow.keras.backend as K


def angular_loss_mae(y_true, y_pred):
    """
    Tensorflow custom loss function to be used during training
    :param y_true:
    :param y_pred:
    :return:
    """

    N = tf.math.divide(y_pred, 360)
    N = tf.floor(N)

    y_pred = K.switch(K.greater_equal(y_pred, 360), y_pred - (N * 360), y_pred)
    y_pred = K.switch(K.less(y_pred, 0), y_pred + (K.abs(N) * 360), y_pred)

    error1 = K.abs(y_pred - y_true)
    error2 = 360 - error1
    error = K.minimum(error1, error2)
    mae = K.mean(error)

    return mae
