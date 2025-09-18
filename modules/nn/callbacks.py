import os
import math

import tensorflow as tf

class Callback():
    def __init__(self):
        self.model = None
        self.trainer = None
    
    def __call__(self, epoch, logs=None):
        self.call(epoch, logs)
    
    def set_model(self, model, trainer):
        self.model = model
        self.trainer = trainer

    def call(self, epoch, logs=None):
        raise NotImplementedError

class SaveCheckpoint(Callback):
    def __init__(self, path, save_period):
        super().__init__()
        self.save_period = save_period

        weights_path = os.path.join(path, "weights")
        self.weights_epoch = os.path.join(weights_path, "epoch-%016d.weights.h5")
        self.weights_best = os.path.join(weights_path, "best.weights.h5")
        self.weights_last = os.path.join(weights_path, "last.weights.h5")
        self.log = os.path.join(path, "train.csv")

        self.best_loss = 1e+100

        if not os.path.isdir(weights_path):
            os.makedirs(weights_path)

    def call(self, epoch, logs=None):
        self.model.save_weights(self.weights_last)

        loss = logs["val_loss"]
        if self.best_loss >= loss:
            self.best_loss = loss
            self.model.save_weights(self.weights_best)

        if self.save_period > 0 and epoch % self.save_period == 0:
            self.model.save_weights(self.weights_epoch % (epoch + 1))
        
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
    def __init__(self, learning_rate=1e-3, scheduler_type="linear", decay_lr=1.0, decay_start=0, decay_epochs=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.decay_lr = decay_lr
        self.decay_start = decay_start - 1
        self.decay_epochs = decay_epochs

        self.decay_step = 0.0
    
    def call(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate.numpy()

        if self.decay_start > 0 and epoch > self.decay_start:
            decay_epoch = epoch - self.decay_start
            if self.scheduler_type == "linear":
                lr = self.learning_rate - (self.learning_rate - self.learning_rate * self.decay_lr) / self.decay_epochs * decay_epoch
            elif self.scheduler_type in ("cos", "cosine"):
                lr = (math.cos(math.pi / self.decay_epochs * decay_epoch) + 1) / 2 * (self.learning_rate - self.learning_rate * self.decay_lr) + self.learning_rate * self.decay_lr
            else:
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        
        self.model.optimizer.learning_rate.assign(lr)

class EarlyStopping(Callback):
    def __init__(self, patience=50):
        super().__init__()
        self.patience = patience
        self.best_loss = 1e+100
        self.best_epoch = 0
    
    def call(self, epoch, logs=None):
        if self.best_loss > logs["val_loss"]:
            self.best_loss = logs["val_loss"]
            self.best_epoch = epoch
        
        if epoch - self.best_epoch >= self.patience:
            self.trainer.stop()