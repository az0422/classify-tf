import tensorflow as tf

from tensorflow.keras.layers import Layer, AveragePooling2D
from tensorflow.keras.models import Sequential

from ..layers import ConvR

class DenseNetBase(Layer):
    def __init__(self, in_channels, gr=32):
        super().__init__()

        self.m = Sequential([
            ConvR(in_channels, 4 * gr, 1, 1),
            ConvR(4 * gr, gr, 3, 1),
        ])
    
    def call(self, x, training=None):
        y = self.m(x, training=training)
        return tf.concat([x, y], axis=-1)

class DenseNet(Layer):
    def __init__(self, in_channels, gr=32, n=1):
        super().__init__()

        m = []
        channels = in_channels
        for _ in range(n):
            m.append(DenseNetBase(channels, gr))
            channels += gr
        
        self.m = Sequential(m)
    
    def call(self, x, training=None):
        return self.m(x, training=training)

class Transition(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = Sequential([
            ConvR(in_channels, out_channels, 1, 1),
            AveragePooling2D(2, 2),
        ])
    
    def call(self, x, training=None):
        return self.m(x, training=training)
