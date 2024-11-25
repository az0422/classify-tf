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
    "SPPF": SPPF,

    "EEBResNet_1": EEBResNet_1,
    "EEBResNet_2": EEBResNet_2,
    "EEBResNet_3": EEBResNet_3,
    "EEBResNet_4": EEBResNet_4,

    "Classify": Classify,
}