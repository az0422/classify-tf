import os

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SaveCheckpoint(Callback):
    def __init__(self, path, save_period):
        super().__init__()
        self.save_period = save_period

        weights_path = os.path.join(path, "weights")
        self.weights = os.path.join(weights_path, "epoch-%016d.ckpt" if tf.__version__ < "2.16.0" else "epoch-%016d.weights.h5")
        self.log = os.path.join(path, "accuracy.txt")

        if not os.path.isdir(weights_path):
            os.makedirs(weights_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_period == 0:
            self.model.save_weights(self.weights % (epoch + 1))
        
        if hasattr(self.model.optimizer, "lr"):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        else:
            lr = self.model.optimizer.learning_rate.numpy()

        with open(self.log, "a") as f:
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.16f\n" % (
                epoch + 1,
                logs["accuracy"],
                logs["loss"],
                logs["val_accuracy"],
                logs["val_loss"],
                lr,
            ))

class Scheduler(Callback):
    def __init__(self, learning_rate=1e-3, warmup_lr=1.0, warmup_epochs=0, decay_ratio=1.0, decay_start=0, decay_epochs=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.decay_ratio = decay_ratio
        self.decay_start = decay_start
        self.decay_epochs = decay_epochs

        self.warmup_step = (learning_rate - learning_rate * warmup_lr) / warmup_epochs
        self.decay_step = 0.0
    
    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.model.optimizer, "lr"):
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        else:
            lr = self.model.optimizer.learning_rate.numpy()

        if epoch < self.warmup_epochs:
            lr = (self.learning_rate - self.learning_rate * self.warmup_lr) / self.warmup_epochs * epoch + self.learning_rate * self.warmup_lr
        elif self.decay_start > 0 and epoch > self.decay_start:
            lr = self.learning_rate - (self.learning_rate - self.learning_rate * self.decay_ratio) / self.decay_epochs * (epoch - self.decay_start)
        
        if hasattr(self.model.optimizer, "lr"):
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        else:
            self.model.optimizer.learning_rate.assign(lr)