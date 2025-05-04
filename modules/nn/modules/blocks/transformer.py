import math

import tensorflow as tf

from tensorflow.keras.layers import Layer, LayerNormalization
from tensorflow.keras.models import Sequential

from ..layers import Conv

class ConvMultiHeadAttention(Layer):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()

        assert out_channels % num_heads == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = out_channels // num_heads

        self.conv1 = Conv(in_channels, out_channels, 1, 1, act=False, bn=False)
        self.conv2 = Conv(in_channels, out_channels, 1, 1, act=False, bn=False)
        self.conv3 = Conv(in_channels, out_channels, 1, 1, act=False, bn=False)
        self.conv4 = Conv(out_channels, out_channels, 1, 1, act=False, bn=False)

        self.norm = LayerNormalization(epsilon=1e-6)
    
    def split_heads(self, x):
        batch, height, width, channels = x.shape
        x = tf.reshape(x, (-1, height * width, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])
    
    def call(self, x, training=None):
        q = self.split_heads(self.conv1(x, training=training))
        k = self.split_heads(self.conv2(x, training=training))
        v = self.split_heads(self.conv3(x, training=training))

        score = q @ tf.transpose(k, [0, 1, 3, 2])
        score = score / tf.sqrt(tf.cast(self.depth, x.dtype))
        weights = tf.nn.softmax(score, axis=-1)
        attention = weights @ v

        attention = tf.transpose(attention, [0, 2, 1, 3])
        batch, hw, heads, depth = attention.shape
        attention = tf.reshape(attention, (-1, int(math.sqrt(float(hw))), int(math.sqrt(float(hw))), self.out_channels))

        out = self.conv4(attention, training=training)
        out = self.norm(out + x, training=training)

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