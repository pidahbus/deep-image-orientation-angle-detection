from models import OAD
from config import TRAIN_DF_PATH, TRAIN_DIR_PATH, VALID_DF_PATH, VALID_DIR_PATH, BATCH_SIZE, MODEL_SAVE_DIR, \
    INIT_CHECKPOINT

def train():
    model = OAD()
    model.fit(TRAIN_DIR_PATH, VALID_DIR_PATH, TRAIN_DF_PATH, VALID_DF_PATH, BATCH_SIZE, MODEL_SAVE_DIR, INIT_CHECKPOINT)


if __name__=="__main__":
    train()