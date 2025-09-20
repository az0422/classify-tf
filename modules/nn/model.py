import yaml
import os
import math
import numpy as np
import gc
import copy

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

from .utils import calc_flops

from . import modules

from .modules import (
    BaseLayer,
    FC,
    Conv,
    ConvT,
    ConvTranspose,
    Shortcut,
    Concat,
    Reshape,

    SEBlock,
    CBAM,
    Inception,
    SPPF,

    ResNet,
    ResNet2L,
    SEResNet,
    ResNetSE,

    CSPResNet,
    CSPResNet2C,
    CSPResNet2L2C,
    CSPResNet2L3C,
    CSPSEResNet,

    Classify,
    ClassifyR,
    ClassifyS,
    ClassifyFC,
    CombineOutput,

    MultiHeadAttentionT,
    PositionalEncodingT,
    ConvFFNNT,
)

def quantize_channels(channels):
    return math.ceil(channels / 8) * 8

def parse_model(cfg, classes, image_size=None, default_act=None):
    assert os.path.isfile(cfg) or os.path.isfile(os.path.join("cfg/models", cfg)), "Configration file of model was not found."

    if not os.path.isfile(cfg):
        cfg = os.path.join("cfg/models", cfg)
    
    with open(cfg, "r") as f:
        cfg = yaml.full_load(f)
        f.seek(0)
        cfg_str = f.read()

    width_multiple = cfg["width_multiple"]
    depth_multiple = cfg["depth_multiple"]
    activation = cfg["activation"] if "activation" in cfg.keys() else None

    if activation is None:
        BaseLayer.default_act = layers.Activation("silu" if default_act is None else default_act)
    else:
        BaseLayer.default_act = layers.Activation(activation)
    
    layers_list = [layers.Input(shape=(image_size, image_size, 3))]
    channels = [3]
    layer_info = []

    print("=" * (1 + 8 + 24 + 12 + 40 + 48))
    print(" %-8s%24s%12s%40s%48s" % ("index", "from", "depth", "layer_name", "args"))
    print("-" * (1 + 8 + 24 + 12 + 40 + 48))

    network = cfg["network"] if "network" in cfg.keys() else cfg["backbone"] + cfg["head"]

    for i_, (index, depth, layer_name, args) in enumerate(network):
        if layer_name.startswith("layers."):
            layer = eval(layer_name)
            layer_name = "tf.keras." + layer_name
        elif layer_name.startswith("tf.keras.layers."):
            layer = eval(layer_name)
        else:
            layer = getattr(modules, layer_name)
        
        depth_ = 0
        args_ = []

        if type(depth) is int:
            depth = math.ceil(depth * depth_multiple)
        elif type(depth) is list or tuple:
            depth = [math.ceil(d * depth_multiple) if type(d) is int else int(d) for d in depth]
        else:
            depth = int(depth)
        
        index_ = None

        if type(index) in (int, str):
            if index == "input":
                index_ = 0
            elif index >= 0:
                index_ = index + 1
            else:
                index_ = index
        else:
            index_ = []
            for ii in index:
                if ii == "input":
                    index_.append(0)
                elif ii >= 0:
                    index_.append(ii + 1)
                else:
                    index_.append(ii)

        for i, arg in enumerate(args):
            try:
                arg = eval(arg)
                args[i] = arg
            except:
                pass

        if layer in (
            FC,
            Conv,
            ConvT,
            ConvTranspose,

            SEBlock,
            CBAM,
            Inception,
            SPPF,

            ResNet,
            ResNet2L,
            SEResNet,
            ResNetSE,

            CSPResNet,
            CSPResNet2C,
            CSPResNet2L2C,
            CSPResNet2L3C,
            CSPSEResNet,

            PositionalEncodingT,
            ConvFFNNT,
        ):
            args.insert(0, channels[index_])
            args[1] = quantize_channels(args[1] * width_multiple)
            channels.append(args[1])

            if layer in (
                CSPResNet,
                CSPResNet2C,
                CSPResNet2L2C,
                CSPResNet2L3C,
                CSPSEResNet,
            ):
                args.insert(2, depth)
                depth_ = depth
                depth = 1
            
            if layer in (ResNetSE,):
                assert type(depth) in (list, tuple), "This layer requires the list or tuple depth value."
                args.insert(2, depth[1])
                depth_ = depth
                depth = depth[0]
            
            if layer in (FC, Conv, ConvTranspose):
                if callable(args[-1]):
                    args_ = copy.copy(args)
                    args_[-1] = "tf.nn." + args[-1].__name__
        
        elif layer in (
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv2DTranspose,
        ):
            args[0] = quantize_channels(args[0] * width_multiple)
            channels.append(args[0])
        
        elif layer in (
            Shortcut,
            tf.keras.layers.Multiply,
            tf.keras.layers.Add
        ):
            channels.append(channels[index_[0]])
        
        elif layer is Concat:
            ch = [channels[i] for i in index_]
            channels.append(sum(ch))
        
        elif layer is Reshape:
            assert type(args[0]) is list or tuple, "This layer requires list of shape."
            ch = args[0][-1]
            channels.append(ch)

        elif layer in (Classify, ClassifyR, ClassifyS, ClassifyFC):
            ch = channels[index_]
            args.insert(0, ch)
            args.insert(1, classes)
            channels.append(classes)
        
        elif layer in (MultiHeadAttentionT,):
            ch = [channels[i] for i in index_]
            args.insert(0, ch)

            channels.append(args[1])
        
        elif layer is CombineOutput:
            ch = channels[index_[0]]
            channels.append(ch)

        else:
            ch = channels[index_]
            channels.append(ch)
            
        print(" %-8s%24s%12s%40s%48s" % (i_, index, depth_ if depth_ else depth, layer_name, args_ if args_ else args))
        
        if depth != 1:
            m = Sequential([layer(*args) for _ in range(depth)])
        else:
            m = layer(*args)
        
        prev = layers_list[index_] if type(index_) is int else [layers_list[i] for i in index_]
        out = m(prev)

        layers_list.append(out)

        layer_info.append([index, depth, layer_name, args])
    
    print("-" * (1 + 8 + 24 + 12 + 40 + 48))

    return layers_list, layer_info, cfg_str

class ClassifyModel(Model):
    def __init__(self, cfg, classes, image_size=None, name="classify", **kwargs):
        self.layers_list, self.layer_info, self.cfg = parse_model(cfg, classes, image_size)

        super().__init__(self.layers_list[0], self.layers_list[-1], name=name, **kwargs)

        print("Total parameters: %.4f M" % (self.count_params() / 1e+6))
        print("Total FLOPs: %.4f GFLOPs per image" % (calc_flops(self) / 1e+9))

    def getConfig(self):
        return self.cfg