import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from ..layers import Conv, ConvR
from .base import CSPBaseS, CSPBaseP

class BaseResNet(Layer):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.m = None

        if strides != 1 or in_channels != out_channels:
            self.ds = Conv(in_channels, out_channels, 1, strides, act=False)
        else:
            self.ds = None
    
    def call(self, x, training=None):
        y = self.m(x, training=training)

        if self.ds is not None:
            return Conv.default_act(y + self.ds(x, training=training))
        return Conv.default_act(y + x)

class BaseResNetV2(Layer):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()

        self.m = None

        if strides != 1 or in_channels != out_channels:
            self.ds = Sequential([
                ConvR(in_channels, out_channels, 1, strides),
            ])
        else:
            self.ds = None
    
    def call(self, x, training=None):
        y = self.m(x, training=training)

        if self.ds is not None:
            return y + self.ds(x, training=training)
        return y + x

class ResNet2L(BaseResNet):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__(in_channels, out_channels, strides)

        self.m = Sequential([
            Conv(in_channels, out_channels, 3, strides),
            Conv(out_channels, out_channels, 3, 1, act=False),
        ])

class ResNet3L(BaseResNet):
    def __init__(self, in_channels, out_channels, strides=1, expand=4):
        super().__init__(in_channels, out_channels, strides)

        channels_h = out_channels // expand
        self.m = Sequential([
            Conv(in_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, 3, strides),
            Conv(channels_h, out_channels, 1, 1, act=False),
        ])

class ResNet2LV2(BaseResNetV2):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__(in_channels, out_channels, strides)

        self.m = Sequential([
            ConvR(in_channels, out_channels, 3, strides),
            ConvR(out_channels, out_channels, 3, 1),
        ])

class ResNet3LV2(BaseResNetV2):
    def __init__(self, in_channels, out_channels, strides=1, expand=4):
        super().__init__(in_channels, out_channels, strides)

        channels_h = out_channels // expand
        self.m = Sequential([
            ConvR(in_channels, channels_h, 1, 1),
            ConvR(channels_h, channels_h, 3, strides),
            ConvR(channels_h, out_channels, 1, 1),
        ])

class CSPResNet2LS(CSPBaseS):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        
        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 1, 1)
        
        self.m = Sequential([
            ResNet2L(out_channels // 2, out_channels // 2) for _ in range(n)
        ])

class CSPResNet3LS(CSPBaseS):
    def __init__(self, in_channels, out_channels, n=1, expand=4):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 1, 1)

        self.m = Sequential([
            ResNet3L(out_channels // 2, out_channels // 2, expand=expand) for _ in range(n)
        ])

class CSPResNet2LP(CSPBaseP):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv2 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv3 = Conv(out_channels, out_channels, 1, 1)

        self.m = Sequential([
            ResNet2L(out_channels // 2, out_channels // 2) for _ in range(n)
        ])

class CSPResNet3LP(CSPBaseP):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv2 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv3 = Conv(out_channels, out_channels, 1, 1)

        self.m = Sequential([
            ResNet3L(out_channels // 2, out_channels // 2) for _ in range(n)
        ])