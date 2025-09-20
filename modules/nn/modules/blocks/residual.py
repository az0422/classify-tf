import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from ..layers import Conv, ConvT
from .base import SEBlock

class ResNet(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5, kernel=3):
        assert in_channels == out_channels
        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            Conv(out_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, kernel, 1),
            Conv(channels_h, out_channels, 1, 1),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)

class ResNet2L(Layer):
    def __init__(self, in_channels, out_channels, expand=1.0, kernel=3):
        assert in_channels == out_channels

        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            Conv(out_channels, channels_h, kernel, 1),
            Conv(channels_h, out_channels, 3, 1),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)

class ResNetSE(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5, ratio=16):
        assert in_channels == out_channels
        super().__init__()

        self.m = Sequential([ResNet(out_channels, out_channels, expand) for _ in range(n)])
        self.se = SEBlock(out_channels, out_channels, ratio)
    
    def call(self, x, training=None):
        x = self.m(x, training=training)
        y = self.se(x, training=training)

        return y

class SEResNet(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5, ratio=16, kernel=3):
        assert in_channels == out_channels
        super().__init__()

        self.m = ResNet(out_channels, out_channels, expand, kernel)
        self.se = SEBlock(out_channels, out_channels, ratio)
    
    def call(self, x, training=None):
        y = self.m(x, training=training)
        y = self.se(y, training=training)

        return x + y

class ResNetT(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5, kernel=3):
        assert in_channels == out_channels
        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            ConvT(out_channels, channels_h, 1, 1),
            ConvT(channels_h, channels_h, kernel, 1),
            ConvT(channels_h, out_channels, 1, 1),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)

class ResNetT2L(Layer):
    def __init__(self, in_channels, out_channels, expand=1.0, kernel=3):
        assert in_channels == out_channels

        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            ConvT(out_channels, channels_h, kernel, 1),
            ConvT(channels_h, out_channels, 3, 1),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)