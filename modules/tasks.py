import yaml
import math
import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import Callback

from .nn import (
    Conv,
    ConvTranspose,
    Concat,
    Shortcut,
    ResNet,
    CSPResNet,
    Bottleneck,
    C2,
    C3,
    EDBlock,
    Classify,
)

def align_channels(channels, width_multiple, unit=8):
    return width_multiple * channels // unit * unit

def parse_model(cfg, nc):
    cfg = yaml.full_load(open(cfg, "r"))

    width_multiple = cfg["width_multiple"]
    depth_multiple = cfg["depth_multiple"]
    activation = eval(cfg["activation"]) if "activation" in cfg.keys() else None

    head = cfg["head"]
    body = cfg["body"]

    layers_ = [layers.Input(shape=[None, None, 3])]

    if activation is not None:
        Conv.default_act = activation

    for index, num, layer, args_str in body + head:
        args = []
        for arg in args_str:
            if type(arg) is str:
                try:
                    arg_t = eval(arg)
                    arg = arg_t
                except:
                    pass

            args.append(arg)
        layer = eval(layer)
        num = math.ceil(num * depth_multiple)

        if layer in (Conv,
                     ConvTranspose,
                     ResNet,
                     CSPResNet,
                     Bottleneck,
                     C2,
                     C3,
                     EDBlock,
        ):
            args[0] = align_channels(args[0], width_multiple, 8)

            if layer in (CSPResNet, C2, C3, EDBlock):
                args.append(num)
                num = 1
        
        elif layer in (Concat, Shortcut):
            pass
        elif layer in (Classify, ):
            if args[0] == "nc":
                args[0] = nc
        
        if num > 1:
            m = Sequential([layer(*args) for _ in range(num)])
        else:
            m = layer(*args)
        m_ = m(layers_[index])
        layers_.append(m_)
    
    return Model(layers_[0], layers_[-1])

class SaveCheckpoint(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

        with open(os.path.join(*(os.path.split(filename)[:-1]), "accuracy.txt"), "w") as f:
            f.write("epoch,train accuracy,train loss,val accuracy,val loss,learing rate\n")

    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(*(os.path.split(self.filename)[:-1]))

        if not os.path.isdir(path):
            os.makedirs(path)
        
        self.model.save_weights(self.filename % (epoch + 1))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        with open(os.path.join(path, "accuracy.txt"), "a") as f:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.16f\n" % (
                epoch + 1,
                logs["accuracy"],
                logs["loss"],
                logs["val_accuracy"],
                logs["val_loss"],
                lr,
            ))
