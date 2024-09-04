import tensorflow as tf

from tensorflow.keras.losses import Loss

class MSE(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        loss = tf.reduce_mean((y_true - y_pred) ** 2)
        return loss

class RMSE(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        loss = tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))
        return loss

class MAE(Loss):
    def __init__(self):
        super().__init__()
    
    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return loss