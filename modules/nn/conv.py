import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Conv2DTranspose

class Conv(Layer):
    default_act = tf.nn.silu

    def __init__(self, channels, kernel, strides=1, padding="same", groups=1, act=default_act):
        super().__init__()
        self.conv = Conv2D(channels, kernel, strides=strides, padding=padding, groups=groups)
        self.bn = BatchNormalization()
        self.act = act
    
    def call(self, x):
        y = self.bn(self.conv(x))
        if self.act is None:
            return y
        return self.act(y)

class ConvTranspose(Layer):
    default_act = tf.nn.silu

    def __init__(self, channels, kernel, strides=1, padding="valid", groups=1, act=default_act):
        super().__init__()
        self.conv = Conv2DTranspose(channels, kernel, strides=strides, padding=padding, groups=groups)
        self.bn = BatchNormalization()
        self.act = act
    
    def call(self, x):
        y = self.bn(self.conv(x))
        if self.act is None:
            return y
        return self.act(y)

class Concat(Layer):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
    
    def call(self, x):
        return tf.concat(x, axis=self.axis)

class Shortcut(Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return x[0] + x[1]
    
