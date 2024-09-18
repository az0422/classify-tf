from .layers import *
from .blocks import *
from .head import *

layers_dict = {
    "Conv": Conv,
    "ConvTranspose": ConvTranspose,
    "Conv1d": Conv1d,
    "Shortcut": Shortcut,
    "Concat": Concat,
    "Reshape": Reshape,
    "ResNet": ResNet,
    "CSPResNet": CSPResNet,
    "SPPF": SPPF,
    "Classify": Classify,
    "Classify1d": Classify1d,
}