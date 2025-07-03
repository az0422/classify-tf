import math

import tensorflow as tf

from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.models import Sequential

from ..layers import Conv

class ConvPositionalEncoding(Layer):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        assert in_channels == out_channels

        self.conv1 = Conv(in_channels, in_channels, [kernel, 1], bn=False)
        self.conv2 = Conv(in_channels, in_channels, [1, kernel], bn=False)
    
    def call(self, x, training=None):
        return x + (self.conv1(x, training=training) + self.conv2(x, training=training))

class ConvMultiHeadAttention(Layer):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()

        if type(out_channels) is int:
            out_channels = [out_channels for _ in range(3)]

        assert len(out_channels) == 3
        assert len(in_channels) == 3
        assert all([out % num_heads == 0 for out in out_channels])

        self.in_channels = in_channels
        self.out_channels = out_channels[2]
        self.num_heads = num_heads
        self.depth = [out // num_heads for out in out_channels]

        self.conv1 = Conv(in_channels[0], out_channels[0], 1, 1, act=False, bn=False)
        self.conv2 = Conv(in_channels[1], out_channels[1], 1, 1, act=False, bn=False)
        self.conv3 = Conv(in_channels[2], out_channels[2], 1, 1, act=False, bn=False)
        self.conv4 = Conv(out_channels[2], out_channels[2], 1, 1, act=False, bn=False)

        self.norm = LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x, depth):
        batch, height, width, channels = x.shape
        x = tf.reshape(x, (-1, height * width, self.num_heads, depth))
        return tf.transpose(x, [0, 2, 1, 3]), height, width
    
    def call(self, x, training=None):
        q, _, _ = self.split_heads(self.conv1(x[0], training=training), self.depth[0])
        k, _, _ = self.split_heads(self.conv2(x[1], training=training), self.depth[1])
        v, h, w = self.split_heads(self.conv3(x[2], training=training), self.depth[2])

        score = q @ tf.transpose(k, [0, 1, 3, 2])
        score = score / tf.sqrt(tf.cast(self.depth[0], x[0].dtype))
        weights = tf.nn.softmax(score, axis=-1)
        attention = weights @ v

        attention = tf.transpose(attention, [0, 2, 1, 3])
        attention = tf.reshape(attention, (-1, h, w, self.out_channels))

        out = self.conv4(attention, training=training)
        out = self.norm(out + x[2], training=training)

        return out

class ConvFFNN(Layer):
    def __init__(self, in_channels, out_channels, ratio=4):
        super().__init__()

        self.conv1 = Conv(in_channels, out_channels * ratio, 1, 1, act=False, bn=False)
        self.conv2 = Conv(out_channels * ratio, out_channels, 1, 1, act=False, bn=False)
        self.norm = LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=None):
        y = self.conv1(x, training=training)
        y = tf.nn.relu(y)
        y = self.conv2(y, training=training)
        y = self.norm(y + x, training=training)

        return y

class ConvPatchPointEmbedding(Layer):
    def __init__(self, in_channels, num_patch=16):
        super().__init__()

        dims = in_channels * (num_patch ** 2)
        self.conv = Conv(in_channels, dims, num_patch, num_patch, padding="valid", act=False, bn=False)
    
    def call(self, x, training=None):
        y = self.conv(x, training=training)

        return y

class ConvTransformer(Layer):
    def __init__(self, in_channels, num_heads=32, ratio=4):
        super().__init__()

        self.m = Sequential([
            ConvMultiHeadAttention(in_channels, in_channels, num_heads),
            ConvFFNN(in_channels, in_channels, ratio),
        ])
    
    def call(self, x, training=None):
        y = self.m(x, training=training)

        return y