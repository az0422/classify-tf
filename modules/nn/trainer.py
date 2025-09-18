import numpy as np
import time

from tqdm import tqdm

import tensorflow as tf

class Trainer():
    def __init__(self, model):
        self.model = model
        self.aux = len(self.model.outputs[0].shape) == 3
        self.aux_length = self.model.outputs[0].shape[1] if self.aux else None

        self.train_begin = []
        self.train_end = []
        self.epoch_begin = []
        self.epoch_begin = []
        self.warmup_epochs = 0
        self.warmup_lr = 0.0

        self.stop_train = False
        self.learning_rate = self.model.optimizer.learning_rate.numpy()
    
    def stop(self):
        self.stop_train = True
    
    def _update_state(self, y_true, y_pred):
        for m in self.model.metrics:
            m.update_state(y_true, y_pred)

    def _print_metrics(self, prefix=""):
        log_str = []
        for m in self.model.metrics:
            result = m.result()
            if type(result) is not dict:
                continue

            for key in result.keys():
                log_str.append("%s: %.4f" % (prefix+key, result[key].numpy()))
        
        return ", ".join(log_str)
    
    def _metrics_to_dict(self, prefix=""):
        log_dict = {}
        for m in self.model.metrics:
            result = m.result()
            if type(result) is not dict: continue

            name, value = tuple(result.items())[0]
            log_dict[prefix+name] = value.numpy()

        return log_dict
    
    @tf.function
    def _apply_gradient(self, gradients, gradient_accumulation_steps):
        grad = [g / gradient_accumulation_steps for g in gradients]
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    @tf.function(jit_compile=True)
    def _train_step(self, x, y, prev_gradients):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = self.model.loss(y, pred)
        
        self._update_state(y, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [g1 + g2 for (g1, g2) in zip(gradients, prev_gradients)]
        return gradients, float(loss)
    
    @tf.function(jit_compile=True)
    def _validate_step(self, x, y):
        pred = self.model(x, training=False)
        loss = self.model.loss(y, pred)

        self._update_state(y, pred)

        return float(loss)
    
    def _train(self, dataloader, epoch, gradient_accumulation_steps=1):
        gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
        losses = []
        dataloader_bar = tqdm(dataloader, mininterval=0.5, bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}")
        start = time.time()

        for it, (x, y) in enumerate(dataloader_bar):
            if len(dataloader) <= it: break

            if epoch < self.warmup_epochs:
                lr = ((self.learning_rate - self.learning_rate * self.warmup_lr) / (self.warmup_epochs * len(dataloader))) * (it + epoch * len(dataloader)) + (self.learning_rate * self.warmup_lr)
                self.model.optimizer.learning_rate.assign(lr)
            
            if self.aux:
                y = tf.expand_dims(y, axis=1)
                y = tf.tile(y, [1, self.aux_length, 1])

            gradients, loss = self._train_step(x, y, gradients)
            losses.append(loss)

            if (it + 1) % gradient_accumulation_steps == 0:
                self._apply_gradient(gradients, gradient_accumulation_steps)
                gradients = [tf.zeros_like(var) for var in self.model.trainable_variables]
            
            if time.time() - start > 0.5:
                dataloader_bar.set_postfix_str(("loss: %.4f, " % (np.mean(losses))) + self._print_metrics(), refresh=False)
                start = time.time()
        
        return np.mean(losses)
    
    def _validate(self, dataloader):
        losses = []
        dataloader_bar = tqdm(dataloader, mininterval=0.5, bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}")
        start = time.time()

        for it, (x, y) in enumerate(dataloader_bar):
            if len(dataloader) <= it: break
            
            loss = self._validate_step(x, y)
            losses.append(loss)
            
            if time.time() - start > 0.5:
                dataloader_bar.set_postfix_str(("val_loss: %.4f, " % (np.mean(losses))) + self._print_metrics("val_"), refresh=False)
                start = time.time()
        
        return np.mean(losses)
    
    def set_callbacks(self, train_begin=[], train_end=[], epoch_begin=[], epoch_end=[], warmup_epochs=0, warmup_lr=0.0):
        self.train_begin = train_begin
        self.train_end = train_end
        self.epoch_begin = epoch_begin
        self.epoch_end = epoch_end
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        for callback in train_begin + train_end + epoch_begin + epoch_end:
            callback.set_model(self.model, self)

    def train(self, dataloader, dataloaderval, epochs=100, gradient_accumulation_steps=1):
        for callback in self.train_begin:
            callback(0, None)
        
        for epoch in range(epochs):
            for callback in self.epoch_begin:
                callback(epoch, None)
            
            print("Epoch %d/%d" % (epoch + 1, epochs))

            train_loss = self._train(dataloader, epoch, gradient_accumulation_steps)
            train_log = self._metrics_to_dict()

            for m in self.model.metrics:
                m.reset_state()
            
            val_loss = self._validate(dataloaderval)
            val_log = self._metrics_to_dict("val_")

            logs = {"loss": train_loss, "val_loss": val_loss}

            for key in train_log.keys():
                logs[key] = train_log[key]
            
            for key in val_log.keys():
                logs[key] = val_log[key]
            
            for callback in self.epoch_end:
                callback(epoch, logs)

            for m in self.model.metrics:
                m.reset_state()

            if self.stop_train:
                break
        
        for callback in self.train_end:
            callback(epoch, logs)