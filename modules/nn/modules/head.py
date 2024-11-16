import math

import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from .layers import Reshape, Conv

class Classify(Layer):
    def __init__(self, in_channels, classes):
        super().__init__()

        channels_h = math.ceil(classes / 8) * 8

        self.m = Sequential([
            Conv(in_channels, channels_h, 3, 1),
            GlobalAveragePooling2D(),
            Reshape([1, 1, channels_h]),
            Conv(channels_h, channels_h, 1, 1),
            Conv(channels_h, classes, 1, 1),
            Reshape([classes]),
            Dense(classes, activation=tf.nn.softmax, dtype=tf.float32)
        ])
    
    def call(self, x):
        y = self.m(x)

        return y
