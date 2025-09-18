from .layers import *
from .blocks import *
from .head import *

layers_dict = {
    "FC": FC,
    "Conv": Conv,
    "ConvT": ConvT,
    "ConvTranspose": ConvTranspose,
    "Shortcut": Shortcut,
    "Concat": Concat,
    "Reshape": Reshape,
    "SizewiseFlatten": SizewiseFlatten,
    "SizewiseDeflatten": SizewiseDeflatten,

    "SEBlock": SEBlock,
    "CBAM": CBAM,
    "Inception": Inception,
    "SPPF": SPPF,

    "ResNet": ResNet,
    "ResNet3L": ResNet,
    "ResNet2L": ResNet2L,
    "SEResNet": SEResNet,
    "ResNetSE": ResNetSE,

    "CSPResNet": CSPResNet,
    "CSPResNet3L3C": CSPResNet,
    "CSPResNet2C": CSPResNet2C,
    "CSPResNet3L2C": CSPResNet2C,
    "CSPSEResNet": CSPSEResNet,
    "CSPSEResNet3L3C": CSPSEResNet,
    "CSPResNet2L3C": CSPResNet2L3C,
    "CSPResNet2L2C": CSPResNet2L2C,

    "Classify": Classify,
    "ClassifyR": ClassifyR,
    "ClassifyS": ClassifyS,
    "ClassifyFC": ClassifyFC,
    "CombineOutput": CombineOutput,

    "PositionalEncodingT": PositionalEncodingT,
    "MultiHeadAttentionT": MultiHeadAttentionT,
}