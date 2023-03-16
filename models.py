from config import VIT_WEIGHTS_PATH
from architectures import create_vit_architecture
import tensorflow as tf

def load_vit_model(load_pretrained_weights=True):
    model = create_vit_architecture()
    if load_pretrained_weights:
        model.load_weights(VIT_WEIGHTS_PATH)
    return model