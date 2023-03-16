import os
from PIL import Image
from models import load_vit_model
import numpy as np
from processing import preprocess, postprocess
from loguru import logger
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class Inference:
    def __init__(self, load_model_name=None):
        logger.info("Loading Models")
        self.vit_model = load_vit_model()

    
    def predict(self, model_name, image_path):
        X = preprocess(model_name, image_path)
        
        y = self.vit_model.predict(X)[0][0]
        
        logger.info(f"Predicted angle is: {y} degree")
        pred_angle = -y
        postprocess(image_path, pred_angle, 400)
        
        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vit")
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()

    model = Inference(load_model_name=args.model_name)
    model.predict(args.model_name, args.image_path)
    
    
