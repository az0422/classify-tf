import os
import cv2
import sys
import numpy as np

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ.keys():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf
from modules.nn.model import ClassifyModel
from modules.dataloader import resize_contain, resize_stretch
from modules.utils import parse_cfg

def resize(cfg, image):
    if cfg["resize_method"] in ("default", "contain"):
        return resize_contain(image, cfg["image_size"])
    elif cfg["resize_method"] in ("stretch",):
        return resize_stretch(image, cfg["image_size"])

def load_model(cfg, path, epoch):
    model_cfg = os.path.join(path, "model.yaml")
    weights_path = os.path.join(path, "weights", epoch + ".keras" if epoch in ("last", "best") else "%016d.keras" % (int(epoch)))
    model = ClassifyModel(model_cfg, cfg["classes"], cfg["image_size"])
    model.load_weights(weights_path)

    return model

def predict(cfg, model, path, image_path):
    if os.path.isfile(image_path):
        images = [image_path]
        out_path = os.path.split(image_path)[0]
    else:
        images = [os.path.join(image_path, img) for img in os.listdir(image_path)]
        out_path = image_path
    
    with open(os.path.join(path, "classes.txt"), "r") as f:
        classes = f.read().split("\n")
    results = []

    for image in images:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = resize(cfg, img)[None, ...].astype(np.float32) // 255.
        pred = model.predict(img, verbose=0)[0]
        pred_cls = np.argmax(pred[0])
        results.append([
            image,
            classes[pred_cls],
        ])
    
    return results, out_path

def out_predict(results, output_format, out_path):
    if output_format == "csv":
        csv = ["%s,%s" % (img, cls) for (img, cls) in results]
        with open(os.path.join(out_path, "result.csv"), "w") as f:
            f.write("\n".join(csv))
        
        print("The result was saved in %s." % (os.path.join(out_path, "result.csv")))

    elif output_format == "stdout":
        print("Prediction results:")
        for img, cls in results:
            print("%s: %s" % (img, cls))

def main(cfg, path, image_path, output_format, epoch):
    tf.keras.mixed_precision.set_global_policy(cfg["mixed_precision"])

    print("Loading model...")
    model = load_model(cfg, path, epoch)

    print("Predicting...")
    pred, out_path = predict(cfg, model, path, image_path)

    out_predict(pred, output_format, out_path)

if __name__ == "__main__":
    path = None
    image_path = None
    output_format = "csv"
    epoch = "best"

    for arg in sys.argv:
        if arg.startswith("weights"):
            path = arg.split("=")[1]
        elif arg.startswith("image"):
            image_path = arg.split("=")[1]
        elif arg.startswith("output_format"):
            output_format = arg.split("=")[1]
        elif arg.startswith("epoch"):
            epoch = arg.split("=")[1]
    
    if epoch in ("last", "best"):
        pass
    else:
        try:
            epoch = int(epoch)
        except:
            print("`epoch` argument should be set `last`, `best`, or integer number.")
            sys.exit()
    
    if path is None:
        print("`weights` argument was not set.")
        sys.exit()
    
    if image_path is None:
        print("`image` argument was not set.")
        sys.exit()
    
    if output_format not in ("csv", "stdout"):
        print("`output_format` should be set only `csv` or `stdout`")
        sys.exit()

    cfg = parse_cfg(os.path.join(path, "cfg.yaml"))

    main(cfg, path, image_path, output_format, epoch)