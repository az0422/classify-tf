import time

import tensorflow as tf

from .utils import getitem, progress_bar

def mean(a):
    return sum(a) / len(a)

class Trainer():
    def __init__(self, model):
        self.model = model
        self.aux = len(self.model.outputs[0].shape) == 3
        self.aux_length = self.model.outputs[0].shape[1] if self.aux else None

        self.train_begin = []
        self.train_end = []
        self.epoch_begin = []
        self.epoch_begin = []

        self.stop_train = False
    
    def _update_state(self, y_true, y_pred):
        for m in self.model.metrics:
            m.update_state(y_true, y_pred)

    def _print_metrics(self, prefix=""):
        log_str = []
        for m in self.model.metrics:
            result = m.result()
            if type(result) is not dict: continue

            name, value = tuple(result.items())[0]
            log_str.append(" %s: %.4f" % (prefix+name, value.numpy()))
        
        return "".join(log_str)
    
    def _metrics_to_dict(self, prefix=""):
        log_dict = {}
        for m in self.model.metrics:
            result = m.result()
            if type(result) is not dict: continue

            name, value = tuple(result.items())[0]
            log_dict[prefix+name] = value.numpy()

        return log_dict
    
    def _print_log(self, loss, log_str, t_start, times, log_len, prefix=""):
        if len(times) > 10:
            del times[0]
        
        log_str.append(" %sms/it" % (round(mean(times) * 1000)))
        log_str.append(" time: %ds" % (round(time.time() - t_start)))
        log_str.append(" %sloss: %.4f" % (prefix, loss))
        log_str.append(self._print_metrics(prefix))
        
        clear = ""
        if log_len is not None:
            clear = " " * log_len
        
        log_str = "".join(log_str)
        print("\r%s\r%s" % (clear, log_str), end="")

        return len(log_str)
    
    @tf.function
    def _new_gradients(self):
        return [tf.zeros_like(var) for var in self.model.trainable_variables]
    
    @tf.function
    def _accumulate_gradient(self, a, b):
        return [g1 + g2 for (g1, g2) in zip(a, b)]
    
    @tf.function(jit_compile=True)
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            loss = self.model.loss(y, pred)
        
        self._update_state(y, pred)
        
        gradient = tape.gradient(loss, self.model.trainable_variables)
        return gradient, float(loss)
    
    @tf.function(jit_compile=True)
    def _apply_gradient(self, gradients, gradient_accumulate_steps):
        grad = [g / gradient_accumulate_steps for g in gradients]
        self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    @tf.function(jit_compile=True)
    def _validate_step(self, x, y):
        pred = self.model(x, training=False)
        loss = self.model.loss(y, pred)

        self._update_state(y, pred)

        return float(loss)
    
    def _train(self, dataloader, gradient_accumulate_steps=1):
        its = len(dataloader)
        gradients = self._new_gradients()
        times = []
        t_start = None
        prev_log_len = None
        losses = []

        for it in range(its):
            log_str = []
            log_str.append(progress_bar(it + 1, its))

            start = time.time()
            if t_start is None:
                t_start = time.time()
            
            x, y = getitem(dataloader)

            if self.aux:
                y = tf.expand_dims(y, axis=1)
                y = tf.tile(y, [1, self.aux_length, 1])

            grad, loss = self._train_step(x, y)
            gradients = self._accumulate_gradient(gradients, grad)
            losses.append(loss)

            if (it + 1) % gradient_accumulate_steps == 0:
                self._apply_gradient(gradients, gradient_accumulate_steps)
                gradients = self._new_gradients()
            
            times.append((time.time() - start))

            if len(times) > 10:
                del times[0]
            
            prev_log_len = self._print_log(mean(losses), log_str, t_start, times, prev_log_len)
        
        return mean(losses)
    
    def _validate(self, dataloader):
        its = len(dataloader)
        times = []
        t_start = None
        prev_log_len = None
        losses = []

        for it in range(its):
            log_str = []
            log_str.append(progress_bar(it + 1, its))

            start = time.time()
            if t_start is None:
                t_start = time.time()
            
            x, y = getitem(dataloader)

            loss = self._validate_step(x, y)
            losses.append(loss)
            times.append((time.time() - start))

            if len(times) > 10:
                del times[0]
            
            prev_log_len = self._print_log(mean(losses), log_str, t_start, times, prev_log_len, "val_")
        
        return mean(losses)
    
    def set_callbacks(self, train_begin=[], train_end=[], epoch_begin=[], epoch_end=[]):
        self.train_begin = train_begin
        self.train_end = train_end
        self.epoch_begin = epoch_begin
        self.epoch_end = epoch_end

        for c in train_begin + train_end + epoch_begin + epoch_end:
            c.set_model(self.model, self)

    def train(self, dataloader, dataloaderval, epochs=100, gradient_accumulation_steps=1):
        for c in self.train_begin:
            c(0, None)
        
        for epoch in range(epochs):
            for c in self.epoch_begin:
                c(epoch, None)
            
            print("Epoch %d/%d" % (epoch + 1, epochs))

            train_loss = self._train(dataloader, gradient_accumulation_steps)
            print("")

            train_log = self._metrics_to_dict()

            for m in self.model.metrics:
                m.reset_state()
            
            val_loss = self._validate(dataloaderval)
            print("")

            logs = {"loss": train_loss, "val_loss": val_loss}

            val_log = self._metrics_to_dict("val_")

            for key in train_log.keys():
                logs[key] = train_log[key]
            
            for key in val_log.keys():
                logs[key] = val_log[key]
            
            for c in self.epoch_end:
                c(epoch, logs)

            for m in self.model.metrics:
                m.reset_state()
            
            if self.stop_train:
                break
        
        for c in self.train_end:
            c(epoch, logs)