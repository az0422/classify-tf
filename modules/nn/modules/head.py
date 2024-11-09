import math

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, GlobalAveragePooling2D, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

from .layers import Reshape, Conv

class Classify(Layer):
    def __init__(self, in_channels, classes):
        super().__init__()
        
        self.gap = GlobalAveragePooling2D()
        self.conv = Sequential([
            Reshape([1, 1, in_channels]),
            Conv(in_channels, in_channels, 1, 1),
            Conv(in_channels, classes, 1, 1),
            Reshape([classes]),
            Dense(classes, activation=tf.nn.softmax, dtype=tf.float32),
        ])
    
    def call(self, x):
        y = self.gap(x)
        y = self.conv(y)

        return y
