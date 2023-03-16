from transformers import TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L


def create_vit_architecture():
    IMAGE_SIZE = 224
    vit_base = TFAutoModel.from_pretrained("google/vit-base-patch16-224")

    img_input = L.Input(shape=(3,IMAGE_SIZE, IMAGE_SIZE))
    x = vit_base(img_input)
    y = L.Dense(1, activation="linear")(x[-1])

    model = Model(img_input, y)
    print(model.summary())
    return model


