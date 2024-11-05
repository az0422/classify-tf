from .layers import *
from .blocks import *
from .head import *

layers_dict = {
    "Conv": Conv,
    "ConvTranspose": ConvTranspose,
    "Shortcut": Shortcut,
    "Concat": Concat,
    "Reshape": Reshape,

    "ResNet": ResNet,
    "CSPResNet": CSPResNet,
    "SPPF": SPPF,

    "Classify": Classify,
}