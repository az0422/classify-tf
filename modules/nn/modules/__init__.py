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

    "SEBlock": SEBlock,
    "CBAM": CBAM,
    "RCAB": RCAB,
    "ResNet": ResNet,
    "ResNetFC": ResNetFC,
    "CSPResNet": CSPResNet,
    "ResNetSE": ResNetSE,
    "Inception": Inception,
    "SPPF": SPPF,

    "Classify": Classify,
}