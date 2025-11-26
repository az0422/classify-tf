import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from ..layers import Conv
from .residual import ResNet, SEResNet

class CSPResNet(Layer):
    def __init__(self, in_channels, out_channels, n=1, expand=0.5):
        super().__init__()

        channels_h = out_channels // 2
        self.conv1 = Conv(in_channels, channels_h, 1, 1)
        self.conv2 = Conv(in_channels, channels_h, 1, 1)
        self.conv3 = Conv(channels_h * 2, out_channels, 1, 1)

        self.m = Sequential([
            ResNet(channels_h, channels_h, expand) for _ in range(n)
        ])
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)
        y1 = self.m(b, training=training)
        y2 = self.conv3(tf.concat([a, y1], axis=-1), training=training)
        return y2

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