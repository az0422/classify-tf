import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Layer, Dense
from tensorflow.keras.models import Sequential

class FC(Layer):
    default_act = [tf.nn.silu]
    def __init__(self, in_nodes, out_nodes, act=True):
        super().__init__()
        self.dense = Dense(out_nodes, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act[0]
        else:
            self.act = None
    
    def call(self, x):
        y = self.bn(self.dense(x))
        if self.act is None:
            return y
        return self.act(y)

class Conv(Layer):
    default_act = [tf.nn.silu]
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="same", groups=1, act=True):
        super().__init__()
        self.conv = Conv2D(out_channels, kernel, strides=strides, padding=padding, groups=groups, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act[0]
        else:
            self.act = None
    
    def call(self, x):
        y = self.bn(self.conv(x))
        if self.act is None:
            return y
        return self.act(y)

class ConvTranspose(Layer):
    default_act = [tf.nn.silu]
    def __init__(self, in_channels, out_channels, kernel, strides=1, padding="valid", act=True):
        super().__init__()
        self.conv = Conv2DTranspose(out_channels, kernel, strides=strides, padding=padding, use_bias=False)
        self.bn = BatchNormalization()

        if type(act) is not bool:
            self.act = act
        elif act:
            self.act = self.default_act[0]
        else:
            self.act = None
    
    def call(self, x):
        y = self.bn(self.conv(x))
        if self.act is None:
            return y
        return self.act(y)

class Shortcut(Layer):
    def __init__(self):
        super().__init__()
    
    def call(self, x):
        return x[0] + x[1]

class Concat(Layer):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis
    
    def call(self, x):
        return tf.concat(x, axis=self.axis)

class Reshape(Layer):
    def __init__(self, shape):
        super().__init__()
        self.reshape_ = shape
    
    def call(self, x):
        batch = tf.shape(x)[0]
        return tf.reshape(x, [batch, *self.reshape_])