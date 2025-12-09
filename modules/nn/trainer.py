import numpy as np
import time

from tqdm import tqdm

import tensorflow as tf

class DataWrapper():
    def __init__(self, data):
        self.data = data
    
    def __next__(self):
        return self.data.__getitem__()

    def __len__(self):
        return len(self.data)

class Trainer():
    def __init__(self, model):
        self.model = model
        self.aux = isinstance(model.output, (list, tuple))
        self.aux_length = len(model.output) if self.aux else None

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

            for name, value in result.items():
                log_dict[prefix+name] = value.numpy().tolist()

        return log_dict

    @tf.function(jit_compile=True)
    def _zero_gradients(self):
        return [tf.zeros_like(var) for var in self.model.trainable_variables]
    
    @tf.function(jit_compile=True)
    def _apply_gradients(self, gradients, gradient_accumulation_steps):
        gradients = [g1 / gradient_accumulation_steps for g1 in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return self._zero_gradients()
    
    @tf.function(jit_compile=True)
    def _train_step(self, x, y, p_gradients):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = self.model.compute_loss(None, y, pred)
        
        n_gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = [g1 + g2 for (g1, g2) in zip(p_gradients, n_gradients)]

        self.model.compute_metrics(None, y, pred)

        return gradients, loss
    
    @tf.function(jit_compile=True)
    def _validate_step(self, x, y):
        pred = self.model(x, training=True)
        loss = self.model.compute_loss(None, y, pred)

        self.model.compute_metrics(None, y, pred)

        return loss
    
    def _train(self, dataloader, epoch, gradient_accumulation_steps=1, log_level="all"):
        losses = []
        gradients = self._zero_gradients()
        dataloader_bar = tqdm(dataloader, mininterval=0.5, bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}")
        data_wrapper = DataWrapper(dataloader)
        it = 0
        start = time.time()

        while it < len(dataloader):
            x, y = next(data_wrapper)
            if not isinstance(y, (tuple, list)):
                y = tuple([y for _ in range(self.aux_length)]) if self.aux_length is not None else y

            gradients, loss = self._train_step(x, y, gradients)
            losses.append(float(loss))

            if epoch < self.warmup_epochs:
                lr = ((self.learning_rate - self.learning_rate * self.warmup_lr) / (self.warmup_epochs * len(dataloader))) * (it + epoch * len(dataloader)) + (self.learning_rate * self.warmup_lr)
                self.model.optimizer.learning_rate.assign(lr)

            if (it  + 1) % gradient_accumulation_steps == 0:
                gradients = self._apply_gradients(gradients, gradient_accumulation_steps)

            if time.time() - start > 0.5:
                dataloader_bar.set_postfix_str(("loss: %.4f, " % (np.mean(losses))) + (self._print_metrics() if log_level == "all" else ""), refresh=False)
                start = time.time()

            dataloader_bar.update(1)
            it += 1
        
        return np.mean(losses)
    
    def _validate(self, dataloader, gradient_accumulation_steps=1, log_level="all"):
        losses = []
        dataloader_bar = tqdm(dataloader, mininterval=0.5, bar_format="{l_bar}{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}{postfix}")
        data_wrapper = DataWrapper(dataloader)
        it = 0
        start = time.time()

        while it < len(dataloader):
            x, y = next(data_wrapper)
            if not isinstance(y, (tuple, list)):
                y = tuple([y for _ in range(self.aux_length)]) if self.aux_length is not None else y

            loss = self._validate_step(x, y)
            losses.append(float(loss))

            if time.time() - start > 0.5:
                dataloader_bar.set_postfix_str(("val_loss: %.4f, " % (np.mean(losses))) + (self._print_metrics("val_") if log_level == "all" else ""), refresh=False)
                start = time.time()
            
            dataloader_bar.update(1)
            it += 1
        
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

    def train(self, dataloader, dataloaderval, epochs=100, gradient_accumulation_steps=1, log_level="all"):
        for callback in self.train_begin:
            callback(0, None)
        
        for epoch in range(epochs):
            for callback in self.epoch_begin:
                callback(epoch, None)
            
            print("Epoch %d/%d" % (epoch + 1, epochs))

            train_loss = self._train(dataloader, epoch, gradient_accumulation_steps, log_level)
            train_log = self._metrics_to_dict()

            for m in self.model.metrics:
                m.reset_state()
            
            val_loss = self._validate(dataloaderval, gradient_accumulation_steps, log_level)
            val_log = self._metrics_to_dict("val_")

            logs = {"loss": train_loss.tolist(), "val_loss": val_loss.tolist()}

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