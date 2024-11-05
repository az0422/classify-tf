import math

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, GlobalAveragePooling2D, Conv1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential

class Classify(Layer):
    def __init__(self, in_channels, classes, act=tf.nn.softmax):
        super().__init__()
        
        self.conv = Conv2D(classes, 3, 1, activation=act)
        self.gap = GlobalAveragePooling2D(dtype="float32")
    
    def call(self, x):
        y = self.conv(x)
        y = self.gap(y)
        return y
