from PIL import Image
import numpy as np
from transformers import ViTFeatureExtractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
from utils import rotate_preserve_size
import os
from config import SAVE_IMAGE_DIR
from loguru import logger
import datetime

def preprocess(model_name, image_path):
    if model_name in ["vit", "tag-cnn"]:
        image_size = 224
    else:
        image_size = 299

    img = Image.open(image_path)
    img = img.resize((image_size, image_size))
    img = np.array(img)
    
    if model_name == "vit":
        X_vit = [img]
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)
        X = X_vit

    if model_name == "tag-cnn":
        X_vit = [img]
        X_vit = feature_extractor(images=X_vit, return_tensors="pt")["pixel_values"]
        X_vit = np.array(X_vit)

        img = np.expand_dims(img, axis=0)
        X = [X_vit, img]

    if model_name in ["efficientnetv2b2", "en", "efficientnetv2b2"]:
        img = np.expand_dims(img, axis=0)
        X = img

    return X
    



def postprocess(img_path, angle, image_size):
    img = rotate_preserve_size(img_path, angle, (image_size, image_size), False)

    # filename = "cs776a-pred.jpg" #img_path.split("/")[-1]
    filename = "pred_" + img_path.split("/")[-1]

    try:
        img.save(os.path.join(SAVE_IMAGE_DIR, filename))
        logger.info(f"Image after orientation angle correction has been saved here: /tmp/{filename}")
    except:
        filename = str(datetime.datetime.now()) + "_" + filename
        img.save(os.path.join(SAVE_IMAGE_DIR, filename))
        logger.info(f"Image after orientation angle correction has been saved here: /tmp/{filename}")