import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D,  GlobalAveragePooling2D, Conv2D

from .layers import *

class AttentionChannel(Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.atn = Sequential([
            Conv(in_channels, in_channels, 3, 1),
            Conv(in_channels, in_channels, 3, 1),
            GlobalAveragePooling2D(),
            Reshape([1, 1, in_channels]),
            Conv(in_channels, in_channels, 1, 1),
            Conv2D(in_channels, 1, 1, activation=tf.nn.sigmoid),
        ])
    
    def call(self, x, training=None):
        return x * self.atn(x, training=training)

class EEB(Layer):
    def __init__(self, in_channels, out_channels, dim):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(in_channels, out_channels, 1, 1)
        self.conv3 = Conv(out_channels, out_channels, 1, 1)

        self.m = Sequential([
            Conv(out_channels, dim, 1, 1),
            GlobalAveragePooling2D(),

            Dense(dim, activation=tf.nn.l2_normalize, use_bias=False),
            FC(dim, out_channels),
            FC(out_channels, out_channels),
            Reshape([1, 1, out_channels]),
        ])
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)

        atn = self.m(a, training=training)

        y = self.conv3(b * atn, training=training)
        return y

class ResNet(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5):
        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            Conv(in_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, 3, 1),
            Conv(channels_h, out_channels, 1, 1),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)

class CSPResNet(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5):
        super().__init__()

        channels_h = out_channels // 2
        self.conv1 = Conv(in_channels, channels_h, 1, 1)
        self.conv2 = Conv(in_channels, channels_h, 1, 1)
        self.conv3 = Conv(channels_h, out_channels, 1, 1)

        self.m = Sequential([
            ResNet(channels_h, channels_h, expand) for _ in range(n)
        ])
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)
        y1 = self.m(b, training=training)
        y2 = self.conv3(tf.concat([a, y1], axis=-1), training=training)
        return y2

class ResNetFC(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5):
        super().__init__()

        channels_h = round(out_channels * expand)

        self.m = Sequential([
            FC(in_channels, channels_h),
            FC(channels_h, channels_h),
            FC(channels_h, out_channels),
        ])
    
    def call(self, x, training=None):
        return x + self.m(x, training=training)

class Inception(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        channels_h = out_channels // 4

        self.conv1 = Conv(in_channels, channels_h, 1, 1)
        self.conv2 = Sequential([
            Conv(in_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, 3, 1),
        ])
        self.conv3 = Sequential([
            Conv(in_channels, channels_h, 1, 1),
            Conv(channels_h, channels_h, 5, 1),
        ])
        self.conv4 = Sequential([
            MaxPooling2D(3, 1, padding="same"),
            Conv(in_channels, channels_h, 1, 1)
        ])
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)
        c = self.conv3(x, training=training)
        d = self.conv4(x, training=training)

        return tf.concat([a, b, c, d], axis=-1)

class SPPF(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = MaxPooling2D(5, 1, padding="same")
        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 1, 1)
    
    def call(self, x, training=None):
        y = [self.conv1(x, training=training)]
        y.extend(self.maxpool(y[-1]) for _ in range(3))
        y = tf.concat(y, axis=-1)
        y = tf.cast(y, dtype=x.dtype)
        return self.conv2(y, training=training)

