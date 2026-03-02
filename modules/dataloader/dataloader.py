import time
import random
import math
import multiprocessing
import copy
import numpy as np

import tensorflow as tf

from multiprocessing import shared_memory

from .utils import load_filelist
from .augment import DataAugment

class DataBuffer():
    def __init__(self, cfg):
        self.cfg = cfg

        self.reader_index_queue = multiprocessing.Queue(cfg["buffer_size"])
        self.writer_index_queue = multiprocessing.Queue(cfg["buffer_size"])

        self.images_buffer = shared_memory.SharedMemory(
            create=True,
            size=cfg["buffer_size"] * cfg["batch_size"] * cfg["image_size"] * cfg["image_size"] * 3
        )

        self.labels_buffer = shared_memory.SharedMemory(
            create=True,
            size=cfg["buffer_size"] * cfg["batch_size"] * cfg["classes"] * 1
        )

        self.images_buffer_np = np.ndarray(
            [
                cfg["buffer_size"],
                cfg["batch_size"],
                cfg["image_size"],
                cfg["image_size"],
                3,
            ],
            dtype=np.uint8,
            buffer=self.images_buffer.buf,
        )

        self.labels_buffer_np = np.ndarray(
            [
                cfg["buffer_size"],
                cfg["batch_size"],
                cfg["classes"],
            ],
            dtype=np.uint8,
            buffer=self.labels_buffer.buf,
        )

        for index in range(cfg["buffer_size"]):
            self.writer_index_queue.put(index)
    
    def writeBuffer(self, images, labels):
        index = self.writer_index_queue.get()

        self.images_buffer_np[index] = images
        self.labels_buffer_np[index] = labels

        self.reader_index_queue.put(index)
    
    def readBuffer(self):
        index = self.reader_index_queue.get()

        images = np.copy(self.images_buffer_np[index])
        labels = np.copy(self.labels_buffer_np[index])

        self.writer_index_queue.put(index)

        return images, labels

    def close(self):
        self.images_buffer.close()
        self.labels_buffer.close()
        self.images_buffer.unlink()
        self.labels_buffer.unlink()

class DataManager(multiprocessing.Process):
    def __init__(self, images, loaders, cfg, val):
        super().__init__()
        random.seed(time.time())

        self.stochastic = cfg["stochastic_ratio"]
        self.batch_size = cfg["batch_size"]
        self.images_index = list(range(len(images)))
        self.queue = multiprocessing.Queue(loaders * 4)

        random.shuffle(self.images_index)

        self.augments = [
            DataAugment(
                random.randrange(-2**31, 2**31 - 1),
                images,
                cfg,
                self.queue,
                not val
            ) for _ in range(loaders)
        ]
        
        self.buffer = DataBuffer(cfg)

        for augment in self.augments:
            augment.set_buffer(self.buffer)
    
    def start(self):
        for augment in self.augments:
            augment.start()
        
        super().start()
    
    def terminate(self):
        for augment in self.augments:
            augment.terminate()
            augment.join()
        
        self.buffer.close()

        super().terminate()
    
    def get(self):
        return self.buffer.readBuffer()
    
    def run(self):
        images = self.images_index
        index = 0

        random.shuffle(images)

        while True:
            if index > len(images) // self.batch_size:
                index = 0
                random.shuffle(images)

            batch = images[index * self.batch_size:(index + 1) * self.batch_size]

            if len(batch) != self.batch_size:
                diff = self.batch_size - len(batch)
                batch.extend(images[:diff])
            
            if self.stochastic > np.random.rand():
                self.queue.put(None)
            else:
                self.queue.put(batch)

            index += 1

class DataLoader():
    def __init__(self, images: str, cfg: dict, val=False, classes_dict=None):
        super().__init__()

        cfg = copy.deepcopy(cfg)
        cfg["batch_size"] = cfg["batch_size"] // cfg["subdivisions"]
        images, classes_dict = load_filelist(images, cfg["file_checkers"], cfg["file_checker_bypass"], classes_dict)
        self.classes_name = list(classes_dict.keys())
        self.classes = len(self.classes_name)
        self.classes_dict = classes_dict
        cfg["classes"] = self.classes

        if type(cfg["data_length"]) is str and cfg["data_length"].startswith("x"):
            self.data_length = math.ceil(len(images) / cfg["batch_size"]) * int(cfg["data_length"][1:])
        elif type(cfg["data_length"]) in (tuple, list):
            self.data_length = cfg["data_length"][int(val)]
        elif cfg["data_length"] is not None and val:
            self.data_length = cfg["data_length"]
        else:
            self.data_length = math.ceil(len(images) / cfg["batch_size"])
        
        if type(cfg["loaders"]) is int:
            loaders = cfg["loaders"]
        elif len(cfg["loaders"]) == 1:
            loaders = cfg["loaders"][0]
        else:
            loaders = cfg["loaders"][int(not val)]
        
        if not val:
            cfg["stochastic_ratio"] = 0.0
        
        self.datamanager = DataManager(images, loaders, cfg, val)

        self.label_smoothing = cfg["label_smoothing"] if not val else 0.0

    def startAugment(self):
        self.datamanager.start()
    
    def stopAugment(self):
        self.datamanager.terminate()
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        images, labels = self.datamanager.get()
        images = tf.convert_to_tensor(images, tf.float32) / 255.
        labels = tf.convert_to_tensor(labels, tf.float32)

        labels = labels * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(self.classes, tf.float32)

        return images, labels