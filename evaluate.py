import os
import sys
import time

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ.keys():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

from tensorflow.keras.losses import CategoricalCrossentropy

from modules.utils import parse_cfg
from modules.nn.model import ClassifyModel
from modules.nn.losses import MSE, MAE, RMSE
from modules.dataloader import DataLoader

def model_compile(cfg, model):
    loss_dict = {"mse": MSE, "rmse": RMSE, "mae": MAE, "cce": CategoricalCrossentropy}
    assert cfg["loss"].lower() in loss_dict.keys(), "Invalid loss function"

    loss = loss_dict[cfg["loss"].lower()]

    model.compile(
        loss=loss(),
        metrics=['accuracy']
    )

    return model

def create_model(cfg):
    model_path = os.path.join(cfg["path"], "model.yaml")
    model = ClassifyModel(model_path, cfg["classes"], cfg["image_size"])

    return model

def load_weights(cfg, model, epoch):
    weights_path = os.path.join(cfg["path"], "weights", (epoch + ".weights.h5" if type(epoch) is str else "epoch-%d.weights.h5" % epoch))

    model.load_weights(weights_path)

    return model

def create_dataloader(cfg):
    dataloader = DataLoader(cfg["val_image"], cfg, False)

    return dataloader

def evaluate_model(model, dataloader):
    def evaluate(x, y):
        pred = model.predict(x, verbose=0)
        loss = model.loss(y, pred)
        model.metrics[-1].update_state(y, pred)

        return loss

    i_time = 0
    loss = 0

    dataloader.startAugment()
    
    for i in range(len(dataloader)):
        x, y = dataloader.__getitem__(i)
        start = time.time()
        loss += evaluate(x, y)
        end = time.time()

        i_time += round((end - start) * 1000)

        print("Evaluate iterations: %d/%d\tspend time: %4dms/it\tloss: %.4f\taccuracy: %.4f" % (
            i + 1,
            len(dataloader),
            i_time // (i + 1),
            loss / (i + 1),
            model.metrics[-1].result()["accuracy"].numpy()
        ), end="\r")

    print()
    dataloader.stopAugment()

    print("loss: %.4f\taccuracy: %.4f" % (
        loss / (i + 1),
        model.metrics[-1].result()["accuracy"].numpy()
    ))

def main(cfg, epoch):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, False)
    
    gpus = tf.config.list_logical_devices('GPU')
    if len(gpus) > 1:
        gpu_process = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print("Detected multi-GPU")
        print([gpu.name for gpu in gpus])
    else:
        gpu_process = tf.distribute.get_strategy()
    
    tf.keras.mixed_precision.set_global_policy(cfg["mixed_precision"])

    print("Load model")
    with gpu_process.scope():
        model = create_model(cfg)
        model = load_weights(cfg, model, epoch)
        model.trainable = False
        model = model_compile(cfg, model)
    
    print("Evaluate Model")
    dataloader = create_dataloader(cfg)
    evaluate_model(model, dataloader)

if __name__ == "__main__":
    image_size = None
    batch_size = None
    epoch = "best"
    path = None

    for arg in sys.argv:
        if arg.startswith("image_size"):
            image_size = int(arg.split("=")[1])
            continue
        if arg.startswith("batch_size"):
            batch_size = int(arg.split("=")[1])
            continue
        if arg.startswith("path"):
            path = arg.split("=")[1]
        if arg.startswith("epoch"):
            if arg.endswith(("best", "last")):
                epoch = arg.split("=")[1]
            else:
                epoch = int(arg.split("=")[1])
    
    if path is None:
        print("Please set the `path` argument!!!")
        sys.exit()
    
    cfg = parse_cfg(os.path.join(path, "cfg.yaml"))

    if image_size is not None:
        cfg["image_size"] = image_size
    
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    
    main(cfg, epoch)