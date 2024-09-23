import os
import sys
import yaml
import cv2
import numpy as np

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ.keys():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, AdamW, Adamax, Ftrl, Lion, Nadam, RMSprop, SGD
from tensorflow.keras.losses import CategoricalCrossentropy

from modules.utils import parse_cfg, apply_local_cfg
from modules.nn.model import ClassifyModel
from modules.nn.callbacks import SaveCheckpoint, Scheduler, GarbageCollect
from modules.nn.losses import MSE, MAE, RMSE
from modules.dataloader import DataLoader

def make_checkpoint_path(cfg):
    checkpoint_path = os.path.join(cfg["checkpoint_path"], cfg["checkpoint_name"])
    if os.path.isdir(checkpoint_path):
        index = 2
        while os.path.isdir(checkpoint_path + str(index)): index += 1
        checkpoint_path = checkpoint_path + str(index)
    
    os.makedirs(checkpoint_path)
    print("checkpoint path:", checkpoint_path)
    return checkpoint_path

def create_model(cfg, checkpoint):
    optimizer_dict = {"adam": Adam, "adadelta": Adadelta, "adagrad": Adagrad, "adamw": AdamW, "adamax": Adamax,
                      "ftrl": Ftrl, "lion": Lion, "nadam": Nadam, "rmsprop": RMSprop, "sgd": SGD}
    loss_dict = {"mse": MSE, "rmse": RMSE, "mae": MAE, "cce": CategoricalCrossentropy}

    assert cfg["optimizer"].lower() in optimizer_dict.keys(), "Invalid optimizer"
    assert cfg["loss"].lower() in loss_dict.keys(), "Invalid loss function"

    classes = len(os.listdir(cfg["train_image"]))

    optimizer = optimizer_dict[cfg["optimizer"].lower()]
    loss = loss_dict[cfg["loss"].lower()]
    learning_rate = cfg["learning_rate"]

    model_cfg = None
    if checkpoint is None:
        model_cfg = cfg["model"]
    else:
        model_cfg = os.path.join(checkpoint, "model.yaml")

    model = ClassifyModel(model_cfg, classes, cfg["image_size"])
    model.build([None, cfg["image_size"], cfg["image_size"], 3])
    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss=loss(),
                  metrics=['accuracy'])
    
    with open(os.path.join(cfg["path"], "model.yaml"), "w") as f:
        f.write(model.getConfig())

    return model, classes

def load_weights(model, checkpoint, epoch):
    weights_file = os.path.join(checkpoint, "weights")
    weights_suffix = ".weights.h5" if tf.__version__ >= "2.16.0" else ".ckpt"

    if epoch in ("last", "best"):
        weights_file = os.path.join(weights_file, epoch) + weights_suffix
    else:
        weights_file = os.path.join(weights_file, "%016d" % (int(epoch)))
    
    model.load_weights(weights_file)

    last_epoch = 0
    train_log = os.path.join(checkpoint, "train.log")

    with open(train_log, "r") as f:
        t = f.read().split("\n")
        last_epoch = len(t)

        if t[-1] == "":
            last_epoch -= 1
    
    if epoch == "best":
        last_epoch = 0
    elif epoch != "last":
        last_epoch = int(epoch)

    return model, last_epoch

def dump_image(images, labels, path, name):
    images = images * 255.
    path = os.path.join(path, "images")

    if not os.path.isdir(path):
        os.makedirs(path)
    
    filename = os.path.join(path, name + "-%d-%08d.png")
    for i, (image, label) in enumerate(zip(images, labels)):
        label = np.argmax(label)
        image = image.astype(np.float32)
        cv2.imwrite(filename % (label, i), image)

def create_dataloaders(cfg):
    dataloader = DataLoader(cfg["train_image"], cfg, True)
    dataloaderval = DataLoader(cfg["val_image"], cfg, False)

    dataloader.startAugment()
    dataloaderval.startAugment()

    checkpoint_path = cfg["path"]

    classes_txt = os.path.join(checkpoint_path, "classes.txt")

    with open(classes_txt, "w") as f:
        f.write("\n".join(dataloader.classes_name))
    
    image, label = dataloader.__getitem__(0)
    dump_image(image.numpy(), label.numpy(), checkpoint_path, "train")
    image, label = dataloaderval.__getitem__(0)
    dump_image(image.numpy(), label.numpy(), checkpoint_path, "val")

    return dataloader, dataloaderval

def train(model, dataloader, dataloaderval, cfg, epoch):
    model.fit(
        dataloader,
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        initial_epoch=epoch,
        validation_data=dataloaderval,
        callbacks=[
            SaveCheckpoint(cfg["path"], cfg["save_period"]),
            Scheduler(
                learning_rate=cfg["learning_rate"],
                warmup_lr=cfg["warmup_lr"],
                warmup_epochs=cfg["warmup_epochs"],
                scheduler_type=cfg["scheduler_type"],
                decay_lr=cfg["decay_lr"],
                decay_start=cfg["decay_start"],
                decay_epochs=cfg["epochs"] - cfg["decay_start"],
            ),
            GarbageCollect()
        ],
    )

def main(cfg, checkpoint, epoch, resume):
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

    if not resume:
        checkpoint_path = make_checkpoint_path(cfg)
        cfg["path"] = checkpoint_path

    last_epoch = 0
    print("create model")
    with gpu_process.scope():
        model, classes = create_model(cfg, checkpoint)

        if checkpoint is not None:
            model, last_epoch = load_weights(model, checkpoint, epoch)
    
    if not resume:
        last_epoch = 0
    
    cfg["classes"] = classes
    yaml.dump(cfg, open(os.path.join(checkpoint_path, "cfg.yaml"), "w"))

    print("create data loaders")
    dataloader, dataloaderval = create_dataloaders(cfg)

    print("train start")
    train(model, dataloader, dataloaderval, cfg, last_epoch)

    dataloader.stopAugment()
    dataloaderval.stopAugment()

if __name__ == "__main__":
    cfg = parse_cfg("cfg/settings.yaml")

    local_cfg = None
    checkpoint = None
    epoch = "best"
    resume = False

    for arg in sys.argv:
        if arg.startswith("cfg"):
            local_cfg = arg.split("=", maxsplit=1)[1]
        elif arg.startswith("checkpoint"):
            checkpoint = arg.split("=", maxsplit=1)[1]
        elif arg.startswith("resume"):
            resume = True
            epoch = "last"
        elif arg.startswith("epoch") and epoch == "best":
            epoch = arg.split("=", maxsplit=1)[1]
    
    if resume:
        cfg = parse_cfg(os.path.join(checkpoint, "cfg.yaml"))
    elif local_cfg is not None:
        cfg = apply_local_cfg(cfg, local_cfg)
    
    main(cfg, checkpoint, epoch, resume)