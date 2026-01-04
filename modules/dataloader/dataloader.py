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
            size=cfg["buffer_size"] * cfg["batch_size"] * cfg["classes"]
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

        images = tf.convert_to_tensor(self.images_buffer_np[index], tf.float32) / 255.
        labels = tf.convert_to_tensor(self.labels_buffer_np[index], tf.float32)

        self.writer_index_queue.put(index)

        return images, labels

    def close(self):
        self.images_buffer.close()
        self.labels_buffer.close()
        self.images_buffer.unlink()
        self.labels_buffer.unlink()

class DataLoader():
    def __init__(self, images: str, cfg: dict, val=False):
        super().__init__()
        random.seed(time.time())

        self.cfg = copy.deepcopy(cfg)
        self.images, classes_name = load_filelist(images, self.cfg["file_checkers"], self.cfg["file_checker_bypass"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        if cfg["data_length"] in (tuple, list):
            self.data_length = cfg["data_length"][int(val)]
        elif cfg["data_length"] is not None and val:
            self.data_length = cfg["data_length"]
        else:
            self.data_length = math.ceil(len(self.images) / cfg["batch_size"])
        
        if type(cfg["loaders"]) is int:
            loaders = cfg["loaders"]
        elif len(cfg["loaders"]) == 1:
            loaders = cfg["loaders"][0]
        else:
            loaders = cfg["loaders"][::-1][int(val)]

        self.augments = [
            DataAugment(
                random.randrange(-2**31, 2**31 - 1),
                self.images,
                cfg,
                not val
            ) for _ in range(loaders)
        ]
        
        self.buffer = DataBuffer(cfg)

        for augment in self.augments:
            augment.set_buffer(self.buffer)

    def startAugment(self):
        for augment in self.augments:
            augment.start()
    
    def stopAugment(self):
        for augment in self.augments:
            augment.terminate()
            augment.join()
        
        self.buffer.close()
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        images, labels = self.buffer.readBuffer()

        return images, labels