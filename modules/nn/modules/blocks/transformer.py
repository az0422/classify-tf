import tensorflow as tf

from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras.models import Sequential

from ..layers import ConvT

class PositionalEncodingT(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels
    
    def build(self, input_shape):
        super().build(input_shape)

        self.pe = self.add_weight(
            shape=input_shape[1:],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="positional_encoding",
        )

    def call(self, x, training=None):
        return x + self.pe

class MultiHeadAttentionT(Layer):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()

        assert out_channels % num_heads == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.depth = out_channels // num_heads
    
    def split_heads(self, x):
        batch, t, channels = x.shape
        x = tf.reshape(x, (-1, t, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])
    
    def call(self, x, training=None):
        q = self.split_heads(x[0])
        k = self.split_heads(x[1])
        v = self.split_heads(x[2])

        score = q @ tf.transpose(k, [0, 1, 3, 2])
        score = score / tf.sqrt(tf.cast(self.depth, x[0].dtype))
        weights = tf.nn.softmax(score, axis=-1)
        attention = weights @ v

        attention = tf.transpose(attention, [0, 2, 1, 3])
        batch, t, heads, depth = attention.shape
        out = tf.reshape(attention, (-1, t, self.out_channels))

        return out

class ConvFFNNT(Layer):
    def __init__(self, in_channels, out_channels, kernels=[3], depth=1, d_out=0.1):
        super().__init__()
        assert in_channels == out_channels

        if len(kernels) == 1:
            kernels = kernels * depth
        
        self.m = Sequential([
            ConvT(out_channels, out_channels, k, 1) for k in kernels
        ])
        self.conv1 = ConvT(out_channels, out_channels, 1, 1)
        self.dropout = Dropout(d_out)
    
    def call(self, x, training=None):
        y = self.m(x, training=training)
        y = self.conv1(y, training=training)

        return self.dropout(y, training=training) + x