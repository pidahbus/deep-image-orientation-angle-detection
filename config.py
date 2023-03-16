import os

ROOT_DIR = os.getcwd()
VIT_WEIGHTS_PATH = "weights/model-vit-ang-loss.h5"

SAVE_IMAGE_DIR = os.path.join("/tmp/")

COCO_TRAIN_DIR = "data/train/"
COCO_VALIDATION_DIR = "data/validation-test"
COCO_VALIDATION_TEST_LABEL_CSV_PATH = "data/validation-test.csv"
BATCH_SIZE = 16