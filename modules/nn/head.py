import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, GlobalAveragePooling2D

class Classify(Layer):
    def __init__(self, nc, multiple=False):
        super().__init__()
        self.conv = Conv2D(nc, 1, 1, activation=tf.nn.sigmoid if multiple else tf.nn.softmax)
        self.gap = GlobalAveragePooling2D()
    
    def call(self, x):
        y = self.conv(x)
        y = self.gap(y)
        return y
