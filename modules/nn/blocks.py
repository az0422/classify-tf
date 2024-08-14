import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from .conv import Conv, ConvTranspose

class ResNet(Layer):
    def __init__(self, channels, expand=0.5, shortcut=True):
        super().__init__()

        channels_h = int(channels * expand)
        self.m = Sequential([
            Conv(channels_h, 1, 1),
            Conv(channels_h, 3, 1),
            Conv(channels, 1, 1),
        ])
        self.shortcut = shortcut
    
    def call(self, x):
        y = self.m(x)
        if self.shortcut:
            return x + y
        return y

class CSPResNet(Layer):
    def __init__(self, channels, expand=0.5, shortcut=True, n=1):
        super().__init__()

        channels_h = channels // 2

        self.conv1 = Conv(channels_h, 1, 1)
        self.conv2 = Conv(channels_h, 1, 1)
        self.conv3 = Conv(channels, 1, 1)

        self.m = Sequential([
            ResNet(channels_h, expand, shortcut) for _ in range(n)
        ])
    
    def call(self, x):
        part_a = self.conv1(x)
        part_b = self.conv2(x)

        y = self.conv3(
            tf.concat([part_a, self.m(part_b)], axis=-1)
        )

        return y

class Bottleneck(Layer):
    def __init__(self, channels, expand=0.5, shortcut=True):
        super().__init__()

        channels_h = int(channels * expand)
        self.m = Sequential([
            Conv(channels_h, 1, 1),
            Conv(channels, 3, 1)
        ])
        self.shortcut = shortcut
    
    def call(self, x):
        y = self.m(x)
        if self.shortcut:
            return x + y
        return y

class C2(Layer):
    def __init__(self, channels, expand=0.5, shortcut=True, n=1):
        super().__init__()

        self.conv1 = Conv(channels, 1, 1)
        self.conv2 = Conv(channels, 1, 1)
        self.m = Sequential([
            Bottleneck(channels, expand, shortcut) for _ in range(n)
        ])
    
    def call(self, x):
        part_a, part_b = tf.split(self.conv1(x), 2, axis=-1)
        y = self.conv2(
            tf.concat([part_a, self.m(part_b)], axis=-1)
        )

        return y

class C3(Layer):
    def __init__(self, channels, expand=0.5, shortcut=True, n=1):
        super().__init__()
        
        channels_h = channels // 2
        self.conv1 = Conv(channels_h, 1, 1)
        self.conv2 = Conv(channels_h, 1, 1)
        self.conv3 = Conv(channels, 1, 1)

        self.m = Sequential([
            Bottleneck(channels_h, expand, shortcut) for _ in range(n)
        ])
    
    def call(self, x):
        part_a = self.conv1(x)
        part_b = self.conv2(x)

        y = self.conv3(
            tf.concat([part_a, self.m(part_b)], axis=-1)
        )

        return y

class EDBlock(Layer):
    def __init__(self, channels, n=1):
        super().__init__()

        channels_h = channels * 2
        self.conv1 = Conv(channels_h, 3, 2)
        self.conv2 = ConvTranspose(channels, 2, 2)
        self.conv3 = Conv(channels, 1, 1)
        self.m = Sequential([
            Conv(channels_h, 3, 1) for _ in range(n)
        ])
    
    def call(self, x):
        y = self.conv1(x)
        y = self.m(y)
        y = self.conv2(y)
        return x + self.conv3(y)
