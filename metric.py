import os
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score

class Accuracy(Callback):

    def __init__(self, batch_size):
        super(Accuracy, self).__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        """
        epoch: Epoch number
        """
        # X, y = self.validation_data
        y_pred = self.model.predict(self.X, batch_size=self.batch_size).reshape(-1, )
        y_true = self.y.reshape(-1, )

        y_pred_class = []
        for y in y_pred:
            if y <= 11.25:
                y_pred_class.append(0.0)
            elif y <= 33.75:
                y_pred_class.append(22.5)
            elif y <= 56.25:
                y_pred_class.append(45)
            elif y <= 78.75:
                y_pred_class.append(67.5)
            elif y <= 101.25:
                y_pred_class.append(90)
            elif y <= 123.75:
                y_pred_class.append(112.5)
            elif y <= 146.25:
                y_pred_class.append(135)
            elif y <= 168.75:
                y_pred_class.append(157.5)
            elif y <= 191.25:
                y_pred_class.append(180)
            elif y <= 213.75:
                y_pred_class.append(202.5)
            elif y <= 236.25:
                y_pred_class.append(225)
            elif y <= 258.75:
                y_pred_class.append(247.5)
            elif y <= 281.25:
                y_pred_class.append(270)
            elif y <= 303.5:
                y_pred_class.append(292.5)
            elif y <= 326:
                y_pred_class.append(315)
            elif y <= 348.5:
                y_pred_class.append(337.5)
            elif y <= 360:
                y_pred_class.append(0.0)
        
        y_true = y_true.astype("str")
        y_pred_class = y_pred_class.astype("str")
        
        acc = accuracy_score(y_true, y_pred_class)

        print(f" - Accuracy: {acc}")