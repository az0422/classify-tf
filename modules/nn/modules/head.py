import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Conv2D, Activation
from tensorflow.keras.models import Sequential

from .layers import Conv, FC

class Classify(Layer):
    def __init__(self, in_channels, classes, act="softmax"):
        super().__init__()

        self.m = Sequential([
            GlobalAveragePooling2D(),
            Dense(classes),
        ])

        self.act = Activation(act.lower()) if act.lower() != "logits" else None
    
    def call(self, x, training=None):
        y = self.m(x, training=training)
        
        if self.act is not None:
            y = self.act(y)

        return y

class ClassifyFC(Layer):
    def __init__(self, in_channels, classes):
        super().__init__()

        self.m = Sequential([
            Dense(classes, activation=tf.nn.softmax, dtype=tf.float32)
        ])
    
    def call(self, x, training=None):
        return self.m(x, training=training)

class Bitmap(Layer):
    def __init__(self, in_channels, channels=3, act="sigmoid"):
        super().__init__()
        
        self.m = Sequential([
            Conv2D(channels, 1, 1, use_bias=False, activation=act, dtype=tf.float32)
        ])
    
    def call(self, x, training=None):
        y = self.m(x, training=training)

        return y

class CombineOutput(Layer):
    def __init__(self, _, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x, training=None):
        return x