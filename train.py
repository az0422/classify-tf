import sys
import os
import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import *
from modules.tasks import parse_model, SaveCheckpoint
from modules.dataloader import DataLoader, DataLoaderVal
from modules.dataloader.utils import load_filelist
from modules.utils import print_help_train

def main():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)
            return

    if "help" in sys.argv:
        print_help_train()
        return
    
    args = yaml.full_load(open("cfg/train.yaml"))

    for arg in sys.argv:
        keys = args.keys()
        for key in keys:
            if not arg.startswith(key): continue
            cast = type(args[key])
            args[key] = cast(arg.split("=")[1])
    
    datasets_path = args["datasets"]
    if os.path.isfile(datasets_path):
        datasets = yaml.full_load(open(datasets_path, "r"))
    else:
        datasets = yaml.full_load(open(os.path.join("cfg/datasets", datasets_path), "r"))
    
    train_data = os.path.join(datasets["path"], datasets["train"])
    val_data = os.path.join(datasets["path"], datasets["val"])

    dataloader = DataLoader(train_data, args["image_size"], args["batch_size"], args["augment_cfg"])
    dataloader_val = DataLoaderVal(val_data, args["image_size"], args["batch_size"])
    
    if not os.path.isdir("runs"):
        os.makedirs("runs")

    save_path = os.path.join("runs", args["name"])
    if os.path.isdir(save_path):
        count = 2
        while True:
            save_path = os.path.join("runs", args["name"] + str(count))
            if os.path.isdir(save_path):
                count +=1
                continue
            os.makedirs(save_path)
            break
    
    save_path = os.path.join(save_path, "epoch-%08d.ckpt")
    
    model_path = args["model"]
    if not os.path.isfile(model_path):
        model_path = os.path.join("cfg/models", model_path)
    
    nc = len(datasets["names"])
    optimizer = eval(args["optimizer"])
    model = parse_model(model_path, nc)
    model.compile(optimizer=optimizer(learning_rate=args["learning_rate"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    dataloader.startAugment()
    model.fit(dataloader,
              epochs=args["epochs"],
              batch_size=args["batch_size"],
              validation_data=dataloader_val,
              callbacks=[SaveCheckpoint(save_path)])

if __name__ == "__main__":
    main()