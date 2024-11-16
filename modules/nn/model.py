import yaml
import os
import math
import numpy as np
import gc

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

from .utils import calc_flops

from .modules import (
    layers_dict,
    FC,
    Conv,
    ConvTranspose,
    Shortcut,
    Concat,
    Reshape,
    EEB,
    CSPEEB,
    ResNet,
    CSPResNet,
    SPPF,
    Classify,
)

def quantize_channels(channels):
    return math.ceil(channels / 8) * 8

def parse_model(cfg, classes, image_size=None):
    assert os.path.isfile(cfg) or os.path.isfile(os.path.join("cfg/models", cfg)), "Configration file of model was not found."

    if not os.path.isfile(cfg):
        cfg = os.path.join("cfg/models", cfg)
    
    with open(cfg, "r") as f:
        cfg = yaml.full_load(f)
        f.seek(0)
        cfg_str = f.read()

    width_multiple = cfg["width_multiple"]
    depth_multiple = cfg["depth_multiple"]
    activation = eval(cfg["activation"]) if "activation" in cfg.keys() else None

    if activation is not None:
        Conv.default_act[0] = activation
        ConvTranspose.default_act[0] = activation
        FC.default_act[0] = activation
    
    layers_list = [layers.Input(shape=(image_size, image_size, 3))]
    channels = [3]
    layer_info = []

    print("=" * (1 + 8 + 24 + 12 + 40 + 40))
    print(" %-8s%24s%12s%40s%40s" % ("index", "from", "depth", "layer_name", "args"))
    print("-" * (1 + 8 + 24 + 12 + 40 + 40))

    for i_, (index, depth, layer_name, args) in enumerate(cfg["backbone"] + cfg["head"]):
        if layer_name.startswith("layers."):
            layer = eval(layer_name)
            layer_name = "tf.keras." + layer_name
        elif layer_name.startswith("tf.keras.layers."):
            layer = eval(layer_name)
        else:
            layer = layers_dict[layer_name]
        
        depth_ = 0
        depth = math.ceil(depth * depth_multiple)

        if type(index) is int:
            index_ = index if index < 0 else index + 1
        else:
            index_ = [(i if i < 0 else i + 1) for i in index]

        for i, arg in enumerate(args):
            try:
                arg = eval(arg)
                args[i] = arg
            except:
                pass

        if layer in (
            FC,
            Conv,
            ConvTranspose,
            ResNet,
            CSPResNet,
            EEB,
            CSPEEB,
            SPPF,
        ):
            args.insert(0, channels[index_])
            args[1] = quantize_channels(args[1] * width_multiple)
            channels.append(args[1])

            if layer in (CSPResNet, CSPEEB):
                args.insert(2, depth)
                depth_ = depth
                depth = 1
        
        elif layer in (
            layers.Conv2D,
            layers.Conv2DTranspose,
        ):
            args[0] = quantize_channels(args[0] * width_multiple)
            channels.append(args[0])
        
        elif layer is Shortcut:
            channels.append(channels[index_[0]])
        
        elif layer is Concat:
            ch = [channels[i] for i in index_]
            channels.append(sum(ch))
        
        elif layer is Reshape:
            ch = args[0][-1]
            channels.append(ch)

        elif layer in (Classify,):
            ch = channels[index_]
            args.insert(0, ch)
            args[1] = classes
            channels.append(classes)
        
        else:
            ch = channels[index_]
            channels.append(ch)
        
        print(" %-8s%24s%12s%40s%40s" % (i_, index, depth_ if depth_ else depth, layer_name, args))
        
        if depth != 1:
            m = Sequential([layer(*args) for _ in range(depth)])
        else:
            m = layer(*args)
        
        prev = layers_list[index_] if type(index_) is int else [layers_list[i] for i in index_]
        out = m(prev)

        layers_list.append(out)

        layer_info.append([index, depth, layer_name, args])
    
    print("-" * (1 + 8 + 24 + 12 + 40 + 40))

    return layers_list, layer_info, cfg_str

class ClassifyModel(Model):
    def __init__(self, cfg, classes, image_size=None, name="classify", *args, **kwargs):
        self.layers_list, self.layer_info, self.cfg = parse_model(cfg, classes, image_size)

        super(ClassifyModel, self).__init__(self.layers_list[0], self.layers_list[-1], name=name, *args, **kwargs)

        print("Total parameters: %.4f M" % (self.count_params() / 1e+6))
        print("Total FLOPs: %.4f GFLOPs per image" % (calc_flops(self) / 1e+9))

    def getConfig(self):
        return self.cfg
