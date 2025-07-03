import tensorflow as tf

from tensorflow.keras.losses import Loss, categorical_crossentropy

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

class AuxiliaryCategoricalCrossEntropy(Loss):
    def __init__(self, weights=[]):
        super().__init__()
        self.weights = tf.expand_dims(tf.convert_to_tensor(weights), axis=-1) if weights != [] else 1.
    
    def call(self, y_true, y_pred):
        y = tf.transpose(y_true, [1, 0, 2])
        p = tf.transpose(y_pred, [1, 0, 2])

        losses = categorical_crossentropy(y, p) * self.weights
        losses = tf.transpose(losses, [1, 0])
        loss = tf.reduce_sum(losses, axis=-1)
        loss = tf.reduce_mean(loss)

        return loss