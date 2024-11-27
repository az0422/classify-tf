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

    "EEB": EEB,
    "EmbedExpandBlock": EEB,
    "ResNet": ResNet,
    "CSPResNet": CSPResNet,
    "Inception": Inception,
    "SPPF": SPPF,

    "Classify": Classify,
}