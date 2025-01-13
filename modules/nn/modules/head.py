import math

import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from .layers import Conv, FC

class Classify(Layer):
    def __init__(self, in_channels, classes):
        super().__init__()

        channels_h = math.ceil(classes / 8) * 8

        self.m = Sequential([
            Conv(in_channels, channels_h, 3, 1),
            GlobalAveragePooling2D(),
            FC(channels_h, channels_h),
            FC(channels_h, classes),
            Dense(classes, activation=tf.nn.softmax, dtype=tf.float32)
        ])
    
    def call(self, x, training=None):
        y = self.m(x, training=training)

        return y

class ClassifyR(Layer):
    def __init__(self, in_channels, classes):
        super().__init__()

        self.m = Sequential([
            Conv(in_channels, in_channels, 1, 1),
            GlobalAveragePooling2D(),
            Dense(classes, activation=tf.nn.softmax, dtype=tf.float32)
        ])
    
    def call(self, x, training=None):
        return self.m(x, training=training)