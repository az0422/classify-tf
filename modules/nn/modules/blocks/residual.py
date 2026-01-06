import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

from ..layers import Conv

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

class CSPResNet(Layer):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()

        channels_h = out_channels // 2
        self.conv1 = Conv(in_channels, channels_h, 1, 1)
        self.conv2 = Conv(in_channels, channels_h, 1, 1)
        self.conv3 = Conv(channels_h * 2, out_channels, 1, 1)

        self.m = Sequential([
            ResNet2L(channels_h, channels_h) for _ in range(n)
        ])
    
    def call(self, x, training=None):
        a = self.conv1(x, training=training)
        b = self.conv2(x, training=training)
        y1 = self.m(b, training=training)
        y2 = self.conv3(tf.concat([a, y1], axis=-1), training=training)
        return y2

class CSPResNet2C(Layer):
    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 1, 1)
        self.m = Sequential([
            ResNet2L(out_channels // 2, out_channels // 2) for _ in range(n)
        ])
    
    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        a, b = tf.split(x, 2, axis=-1)

        return self.conv2(tf.concat([self.m(a, training=training), b], axis=-1))