import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D, Conv2DTranspose, BatchNormalization, Layer, Dense
from tensorflow.keras.models import Sequential

class FC(Layer):
    default_act = None
    def __init__(self, in_nodes, out_nodes, act=True):
        super().__init__()
        self.in_nodes = in_nodes
        self.dense = Dense(out_nodes, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act
        else:
            self.act = None
    
    def build(self, input_shape):
        assert input_shape[-1] == self.in_nodes
        super().build(input_shape)
    
    def call(self, x, training=None):
        y = self.bn(self.dense(x), training=training)
        if self.act is None:
            return y
        return self.act(y)

class Conv(Layer):
    default_act = None
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="same", groups=1, act=True):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.conv = Conv2D(out_channels, kernel, strides=strides, padding=padding, groups=groups, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act
        else:
            self.act = None
    
    def build(self, input_shape):
        assert input_shape[-1] == self.in_channels
        super().build(input_shape)
    
    def call(self, x, training=None):
        y = self.bn(self.conv(x), training=training)
        if self.act is None:
            return y
        return self.act(y)

class ConvTranspose(Layer):
    default_act = None
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="valid", act=True):
        super().__init__()
        self.in_channels = in_channels
        self.conv = Conv2DTranspose(out_channels, kernel, strides=strides, padding=padding, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act
        else:
            self.act = None
    
    def build(self, input_shape):
        assert input_shape[-1] == self.in_channels
        super().build(input_shape)
    
    def call(self, x, training=None):
        y = self.bn(self.conv(x), training=training)
        if self.act is None:
            return y
        return self.act(y)

class TemporalConv(Layer):
    default_act = None
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="same", groups=1, act=True):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.conv = Conv1D(out_channels, kernel, strides=strides, padding=padding, groups=groups, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act
        else:
            self.act = None
    
    def build(self, input_shape):
        assert input_shape[-1] == self.in_channels
        super().build(input_shape)
    
    def call(self, x, training=None):
        y = self.bn(self.conv(x), training=training)
        if self.act is None:
            return y
        return self.act(y)

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