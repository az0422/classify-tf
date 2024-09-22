import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, GlobalAveragePooling2D, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential

from .layers import Conv

class Classify(Layer):
    def __init__(self, in_channels, classes, act=tf.nn.softmax):
        super().__init__()
        
        self.conv = Conv2D(classes, 3, 1, activation=act)
        self.gap = GlobalAveragePooling2D(dtype="float32")
    
    def call(self, x):
        y = self.conv(x)
        y = self.gap(y)
        return y

class Classify1d(Layer):
    def __init__(self, in_channels, classes, act=tf.nn.softmax):
        super().__init__()

        self.conv = Conv1D(classes, 3, 1, activation=act, dtype="float32")
        self.gap = GlobalAveragePooling1D(dtype="float32")
    
    def call(self, x):
        y = self.conv(x)
        y = self.gap(y)
        return y