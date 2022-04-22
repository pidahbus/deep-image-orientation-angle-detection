import tensorflow as tf
import tensorflow.keras.backend as K


def angular_loss_mae(y_true, y_pred):
    """
    Tensorflow custom loss function to be used during training
    :param y_true:
    :param y_pred:
    :return:
    """

    error1 = K.abs(y_pred - y_true)
    error2 = 360 - error1
    error = K.minimum(error1, error2)
    mae = K.mean(error)

    return mae
