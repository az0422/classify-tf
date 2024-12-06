from .layers import *
from .blocks import *
from .head import *

layers_dict = {
    "FC": FC,
    "Conv": Conv,
    "ConvTranspose": ConvTranspose,
    "Shortcut": Shortcut,
    "Multiply": Multiply,
    "Concat": Concat,
    "Reshape": Reshape,

    "EEB": EEB,
    "EmbedExpandBlock": EEB,
    "ResNet": ResNet,
    "ResNetFC": ResNetFC,
    "CSPResNet": CSPResNet,
    "Inception": Inception,
    "SPPF": SPPF,

    "Classify": Classify,
}