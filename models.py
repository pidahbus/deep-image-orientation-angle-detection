from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.applications import Xception
from loss import angular_loss_mae
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from loguru import logger


class OAD:
    def _create_model(self):
        conv_base = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
        for layer in conv_base.layers:
            layer.trainable = False

        img = L.Input(shape=(299, 299, 3))
        x = conv_base(img)
        x = L.Flatten()(x)
        x = L.Dense(512, activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Dense(256, activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Dense(64, activation="relu")(x)
        x = L.BatchNormalization()(x)
        y = L.Dense(1, activation="linear")(x)
        model = Model(img, y)
        print(model.summary())
        return model

    def _prepare_generator(self, df_path, dir_path, batch_size):
        datagen = ImageDataGenerator(rescale=1. / 255)
        df = pd.read_csv(df_path)
        df["angle"] = df["angle"].astype("float32")
        generator = datagen.flow_from_dataframe(dataframe=df, directory=dir_path, x_col="image", y_col="angle",
                                                class_mode="raw", batch_size=batch_size, target_size=(299, 299))
        return generator

    def load_model(self, checkpoint):
        """

        :param checkpoint: saved model path
        """
        self.model =tf.keras.models.load_model(checkpoint, custom_objects={"angular_loss_mae":angular_loss_mae})


    def fit(self, train_dir_path, valid_dir_path, train_df_path, valid_df_path,
            batch_size, model_save_dir, initial_checkpoint=None):
        """
        Train Model
        :param train_dir_path: directory path of train images
        :param valid_dir_path: directory path of validation images
        :param train_df_path: csv path of train images
        :param valid_df_path: csv path of validation images
        :param batch_size: train batch size
        :param model_save_dir: directory where model weights will be saved
        :param initial_checkpoint: initial model weights path. If set None model will start training from the beginning.
        """
        if initial_checkpoint:
            logger.info("Loading Model from initial checkpoint")
            self.load_model(initial_checkpoint)
            epoch = int(initial_checkpoint.split("/")[-1].replace(".h5", ""))

        else:
            logger.info("Building Model")
            self.model = self._create_model()
            epoch = 0

        logger.info("Compiling Model")
        self.model.compile(optimizer="adadelta", loss=angular_loss_mae)

        logger.info("Building Generators")
        train_gen = self._prepare_generator(train_df_path, train_dir_path, batch_size)
        valid_gen = self._prepare_generator(valid_df_path, valid_dir_path, batch_size)

        early_stop = EarlyStopping(monitor="val_loss", mode="min", restore_best_weights=True, patience=10)
        checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_dir, "{epoch}.h5"), save_weights_only=False)
        logs = CSVLogger(os.path.join(model_save_dir, "logs.csv"))
        reduce_lr = ReduceLROnPlateau(factor=0.5, monitor="val_loss", mode="min", patience=5)

        logger.info("Training Started")
        self.model.fit(train_gen, epochs=3000, validation_data=valid_gen,
                       callbacks=[early_stop, checkpoint, logs, reduce_lr], initial_epoch=epoch)

    def predict(self, test_dir_path, test_df_path, batch_size):
        """
        Predicts angle output
        :param test_dir_path: directory path of test images
        :param test_df_path: csv path of test images
        :param batch_size: prediction batch size
        :return: prediction angle as array
        """
        logger.info("Prediction Started")
        test_gen = self._prepare_generator(test_df_path, test_dir_path, batch_size)
        preds = self.model.predict(test_gen, batch_size=batch_size)
        return preds
