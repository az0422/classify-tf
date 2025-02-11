import sys
import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ.keys():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"

import tensorflow as tf

from modules.nn.model import ClassifyModel
from modules.utils import parse_cfg

def load_cfg(path):
    cfg_path = os.path.join(path, "cfg.yaml")
    cfg = parse_cfg(cfg_path)

    return cfg

def load_model(path, cfg, image_size):
    cfg_path = os.path.join(path, "model.yaml")
    model = ClassifyModel(cfg_path, cfg["classes"], image_size)

    return model

def load_weights(model, path, epoch):
    if epoch in ("last", "best"):
        weights_path = os.path.join(path, "weights/%s.keras" % epoch)
    else:
        weights_path = os.path.join(path, "weights/epoch-%s.keras" % epoch)
    
    model.load_weights(weights_path)

    return model

def make_dirs(path):
    export_path = os.path.join(path, "export")
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    
    return export_path

def conv_saved_model(path, model):
    export = os.path.join(path, "saved_model")
    tf.saved_model.save(model, export)

def conv_tflite(path, model):
    export_fp32 = os.path.join(path, "tflite_fp32.tflite")
    export_fp16 = os.path.join(path, "tflite_fp16.tflite")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_fp32 = converter.convert()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_fp16 = converter.convert()

    with open(export_fp32, "bw") as f:
        f.write(tflite_model_fp32)

    with open(export_fp16, "bw") as f:
        f.write(tflite_model_fp16)

def main(path, epoch, image_size):
    print("Load model...")
    cfg = load_cfg(path)
    model = load_model(path, cfg, image_size)

    print("Load weights...")
    model = load_weights(model, path, epoch)
    export = make_dirs(path)

    print("Convert SavedModel...")
    conv_saved_model(export, model)

    print("Convert TFLite...")
    conv_tflite(export, model)

    print("Finished.\nSaved at", export)

if __name__ == "__main__":
    path = "runs/train"
    epoch = "best"
    image_size = 224

    for arg in sys.argv:
        if arg.startswith("path"):
            path = arg.split("=")[1]
            continue
        if arg.startswith("epoch"):
            epoch = arg.split("=")[1]
            continue
        if arg.startswith("image_size"):
            image_size = arg.split("=")[1]
    
    if not os.path.isdir(path):
        print("invalid path:", path)
        sys.exit()
    
    main(path, epoch, image_size)
    
