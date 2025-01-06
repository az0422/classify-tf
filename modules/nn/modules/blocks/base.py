import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential

from ..layers import FC, Conv, Reshape

class SEBlock(Layer):
    def __init__(self, in_channels, out_channels, ratio=16):
        super().__init__()

        squeeze_nodes = out_channels // ratio
        self.m = Sequential([
            GlobalAveragePooling2D(),
            FC(in_channels, squeeze_nodes),
            Dense(out_channels, use_bias=False),
            Reshape([1, 1, out_channels])
        ])
    
    def call(self, x):
        atn = self.m(x)
        atn = tf.nn.sigmoid(atn)
        return atn * x

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

        ch_atn = self.reshape((gap + gmp) / 2)
        sp_atn = self.conv2(tf.concat([ap, mp], axis=-1))

        return x * ch_atn * sp_atn

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