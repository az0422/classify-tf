import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D

from .layers import *

class SEBlock(Layer):
    def __init__(self, in_channels, out_channels, ratio=16):
        super().__init__()

        squeeze_nodes = out_channels // ratio
        self.m = Sequential([
            GlobalAveragePooling2D(),
            FC(in_channels, squeeze_nodes),
            FC(squeeze_nodes, out_channels),
            Dense(out_channels, use_bias=False, activation=tf.nn.sigmoid),
            Reshape([1, 1, out_channels])
        ])
    
    def call(self, x):
        return self.m(x) * x

class CBAM(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, 3, 1)
        self.conv2 = Conv2D(1, 7, 1, padding="same", activation=tf.nn.sigmoid)

        self.fc1 = Sequential([
            FC(out_channels, out_channels // 2),
            Dense(out_channels, activation=tf.nn.sigmoid)
        ])

        self.fc2 = Sequential([
            FC(out_channels, out_channels // 2),
            Dense(out_channels, activation=tf.nn.sigmoid)
        ])

        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()

        self.reshape = Reshape([1, 1, out_channels])
    
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        gap = self.fc1(self.gap(x), training=training)
        gmp = self.fc2(self.gmp(x), training=training)
        ap = tf.reduce_mean(x, axis=-1, keepdims=True)
        mp = tf.reduce_max(x, axis=-1, keepdims=True)

        ch_atn = self.reshape(tf.nn.sigmoid(gap + gmp))
        sp_atn = self.conv2(tf.concat([ap, mp], axis=-1))

        return x * ch_atn * sp_atn

class ResNet(Layer):
    def __init__(self, in_channels, out_channels, expand=0.5):
        assert in_channels == out_channels
        super().__init__()
        channels_h = round(out_channels * expand)
        self.m = Sequential([
            Conv(out_channels, channels_h, 1, 1),
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

class CSPSEResNet(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5, ratio=16, kernel=3):
        super().__init__()

        self.m = Sequential([SEResNet(out_channels // 2, out_channels // 2, expand, ratio, kernel) for _ in range(n)])
        self.conv1 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv2 = Conv(in_channels, out_channels // 2, 1, 1)
        self.conv3 = Conv(out_channels, out_channels, 1, 1)
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)

        y1 = self.m(a, training=training)
        y = tf.concat([b, y1], axis=-1)

        y = self.conv3(y)

        return y

class CSPResNet2C(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5, kernel=3):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 1, 1)
        self.m = Sequential([
            ResNet(out_channels // 2, out_channels // 2, expand, kernel) for _ in range(n)
        ])
    
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        a, b = tf.split(x, 2, axis=-1)

        return self.conv2(tf.concat([self.m(a, training=training), b], axis=-1))

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

