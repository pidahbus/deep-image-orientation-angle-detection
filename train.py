from models import load_vit_model
from loss import angular_loss_mae
from generator import RotGenerator, ValidationTestGenerator, ViTRotGenerator, ViTValidationTestGenerator
from tensorflow.keras.optimizers import Adadelta
from config import COCO_TRAIN_DIR, COCO_VALIDATION_DIR, COCO_VALIDATION_TEST_LABEL_CSV_PATH, BATCH_SIZE
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



def train_vit_model(train_dir, val_dir, val_label_csv_path, weights_save_path, batch_size, load_pretrained_weights):
    IMAGE_SIZE = 224
    model = load_vit_model(load_pretrained_weights)
    model.compile(loss=angular_loss_mae, optimizer=Adadelta(learning_rate=0.1))

    train_gen = RotGenerator(train_dir, batch_size, IMAGE_SIZE, is_vit=True)
    val_gen = ValidationTestGenerator(image_dir=val_dir, 
                                    df_label_path=val_label_csv_path,
                                    batch_size=batch_size, dim=IMAGE_SIZE, mode="valid", is_vit=True)
    cp = ModelCheckpoint(weights_save_path, save_weights_only=False, 
                        save_best_only=True, monitor="val_loss")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    es = EarlyStopping(monitor="val_loss", patience=5)
    model.fit(train_gen, validation_data=val_gen, epochs=10000, callbacks=[cp, es, reduce_lr])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--model-name", type=str, default="vit")
    parser.add_argument("--train-dir", type=str, default=COCO_TRAIN_DIR)
    parser.add_argument("--validation-dir", type=str, default=COCO_VALIDATION_DIR)
    parser.add_argument("--validation-label-path", type=str, default=COCO_VALIDATION_TEST_LABEL_CSV_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--weights-save-path", type=str, required=True)
    parser.add_argument("--load-pretrained-weights", type=bool, default=True)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.model_name == "vit":
        train_vit_model(train_dir=args.train_dir, val_dir=args.validation_dir, val_label_csv_path=args.validation_label_path, 
        batch_size=args.batch_size, weights_save_path=args.weights_save_path, load_pretrained_weights=args.load_pretrained_weights)