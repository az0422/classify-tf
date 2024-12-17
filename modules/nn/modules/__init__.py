from .layers import *
from .blocks import *
from .head import *

layers_dict = {
    "FC": FC,
    "Conv": Conv,
    "ConvTranspose": ConvTranspose,
    "Shortcut": Shortcut,
    "Concat": Concat,
    "Reshape": Reshape,

    "ResNet": ResNet,
    "ResNetFC": ResNetFC,
    "CSPResNet": CSPResNet,
    "ResNetEDFC": ResNetEDFC,
    "Inception": Inception,
    "SPPF": SPPF,

    "Classify": Classify,
}