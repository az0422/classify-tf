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
    "ResNet": ResNet,
    "ResNetFC": ResNetFC,
    "CSPResNet": CSPResNet,
    "CSPResNet3C": CSPResNet,
    "ResNetSE": ResNetSE,
    "SEResNet": SEResNet,
    "CSPSEResNet": CSPSEResNet,
    "CSPResNet2C": CSPResNet2C,
    "Inception": Inception,
    "SPPF": SPPF,

    "Classify": Classify,
}