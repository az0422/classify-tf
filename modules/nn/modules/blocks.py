import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, GlobalAveragePooling2D, UpSampling2D

from .layers import *

class ResNet(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5):
        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            Conv(in_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, 3, 1),
            Conv(channels_h, out_channels, 1, 1),
        ])
    
    def call(self, x):
        return x + tf.cast(self.m(x), dtype=x.dtype)

class CSPResNet(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5):
        super().__init__()

        channels_h = out_channels // 2
        self.conv1 = Conv(in_channels, channels_h, 1, 1)
        self.conv2 = Conv(channels_h, channels_h, 1, 1)
        self.conv3 = Conv(channels_h, out_channels, 1, 1)

        self.m = Sequential([
            ResNet(in_channels, channels_h, expand) for _ in range(n)
        ])
    
    def call(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        y1 = tf.cast(self.m(b), dtype=x.dtype)
        y2 = self.conv3(tf.concat([a, y1], axis=-1))
        return y2

class SPPF(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = MaxPooling2D(5, 1, padding="same")
        self.conv1 = Conv(out_channels, 1, 1)
        self.conv2 = Conv(out_channels, 1, 1)
    
    def call(self, x):
        y = [self.conv1(x)]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        y = tf.concat(y, axis=-1)
        y = tf.cast(y, dtype=x.dtype)
        return self.conv2(y)
