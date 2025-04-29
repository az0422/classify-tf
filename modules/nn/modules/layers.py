import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Layer, Dense, Activation
from tensorflow.keras.models import Sequential

class BaseLayer(Layer):
    default_act = None
    def __init__(self, in_channels, act=True, bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.m = None
        self.bn = BatchNormalization() if bn else None

        if type(act) is str:
            self.act = Activation(act)
        elif act:
            self.act = self.default_act
        else:
            self.act = None
    
    def build(self, input_shape):
        assert input_shape[-1] == self.in_channels
        super().build(input_shape)
    
    def call(self, x, training=None):
        if self.m is None:
            raise NotImplementedError
        
        y = self.m(x, training=training)

        if self.bn is not None:
            y = self.bn(y, training=training)

        if self.act is None:
            return y
        return self.act(y)

class FC(BaseLayer):
    def __init__(self, in_channels, out_channels, act=True, bn=True):
        super().__init__(in_channels, act, bn)
        self.m = Dense(out_channels)

class Conv(BaseLayer):
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="same", dilations=1, act=True, bn=True):
        super().__init__(in_channels, act, bn)
        self.m = Conv2D(out_channels, kernel, strides, padding=padding, dilation_rate=dilations, use_bias=False)

class ConvTranspose(BaseLayer):
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="same", act=True, bn=True):
        super().__init__(in_channels, act, bn)
        self.m = Conv2DTranspose(out_channels, kernel, strides, padding=padding, use_bias=False)

class Shortcut(Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x, training=None):
        return x[0] + x[1]

class ImmediateMultiply(Layer):
    def __init__(self, immediate=1.0):
        super().__init__()
        self.immediate = immediate
    
    def call(self, x, training=None):
        return x * self.immediate

class Concat(Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
    
    def call(self, x, training=None):
        return tf.concat(x, axis=self.axis)

class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.reshape_ = shape
    
    def call(self, x, training=None):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch, *self.reshape_])
