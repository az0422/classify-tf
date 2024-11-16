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
    "CSPEEB": CSPEEB,
    "CSPEmbedExpandBlock": CSPEEB,
    "ResNet": ResNet,
    "CSPResNet": CSPResNet,
    "SPPF": SPPF,

    "Classify": Classify,
}