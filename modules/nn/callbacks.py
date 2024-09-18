import os
import gc
import math

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SaveCheckpoint(Callback):
    def __init__(self, path, save_period):
        super().__init__()
        self.save_period = save_period

        weights_path = os.path.join(path, "weights")
        suffix = ".weights.h5" if tf.__version__ >= "2.16.0" else ".ckpt"
        self.weights_epoch = os.path.join(weights_path, "epoch-%016d") + suffix
        self.weights_best = os.path.join(weights_path, "best") + suffix
        self.weights_last = os.path.join(weights_path, "last") + suffix
        self.log = os.path.join(path, "train.csv")

        self.best_accuracy = 0

        if not os.path.isdir(weights_path):
            os.makedirs(weights_path)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.weights_last)

        accuracy = logs["val_accuracy"]
        if self.best_accuracy <= accuracy:
            self.best_accuracy = accuracy
            self.model.save_weights(self.weights_best)

        if self.save_period > 0 and epoch % self.save_period == 0:
            self.model.save_weights(self.weights_epoch % (epoch + 1))
        
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
    def __init__(self, learning_rate=1e-3, warmup_lr=1.0, warmup_epochs=0, scheduler_type="linear", decay_lr=1.0, decay_start=0, decay_epochs=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.decay_lr = decay_lr
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
            decay_epoch = epoch - self.decay_start
            if self.scheduler_type == "linear":
                lr = self.learning_rate - (self.learning_rate - self.learning_rate * self.decay_lr) / self.decay_epochs * decay_epoch
            elif self.scheduler_type == "cos":
                lr = (math.cos(math.pi / self.decay_epochs * decay_epoch) + 1) / 2 * (self.learning_rate - self.learning_rate * self.decay_lr) + self.learning_rate * self.decay_lr
            else:
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        
        if hasattr(self.model.optimizer, "lr"):
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        else:
            self.model.optimizer.learning_rate.assign(lr)

class GarbageCollect(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, *args, **kwargs):
        gc.collect()