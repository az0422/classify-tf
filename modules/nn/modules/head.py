import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from .layers import Conv

class Classify(Layer):
    def __init__(self, in_channels, classes, act=tf.nn.softmax):
        super().__init__()
        if type(in_channels) is not int:
            self.conv1 = [
                Sequential([Conv(ch, 3, 1), GlobalAveragePooling2D()]) for ch in in_channels
            ]
            self.dense1 = Dense(sum(in_channels), activation=Conv.default_act[0])
            self.dense2 = Dense(classes, activation=act, dtype="float32")
        else:
            self.conv1 = [Sequential([Conv(in_channels, 3, 1), GlobalAveragePooling2D()])]
            self.dense1 = Dense(in_channels, activation=Conv.default_act[0])
            self.dense2 = Dense(classes, activation=act, dtype="float32")
        
    def call(self, x):
        y = [conv(xx) for (conv, xx) in zip(self.conv1, x)]
        y = tf.concat(y, axis=-1)

        y = self.dense1(y)
        y = self.dense2(y)

        return y