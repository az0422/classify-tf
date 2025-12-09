import time
import random
import math
import numpy as np
import multiprocessing
import copy

import tensorflow as tf

from multiprocessing import shared_memory

from .utils import load_filelist
from .augment import DataAugment

class DataBuffer():
    def __init__(self, cfg):
        self.reader_index_queue = multiprocessing.Queue(cfg["buffer_size"])
        self.writer_index_queue = multiprocessing.Queue(cfg["buffer_size"])

        self.buff_images = shared_memory.SharedMemory(
            create=True,
            size=cfg["buffer_size"] * cfg["batch_size"] * (cfg["image_size"] ** 2) * 3
        )

        self.buff_labels = shared_memory.SharedMemory(
            create=True,
            size=cfg["buffer_size"] * cfg["batch_size"] * cfg["classes"]
        )

        self.buff_images_np = np.ndarray(
            [
                cfg["buffer_size"],
                cfg["batch_size"],
                cfg["image_size"],
                cfg["image_size"],
                3,
            ],
            dtype=np.uint8,
            buffer=self.buff_images.buf
        )

        self.buff_labels_np = np.ndarray(
            [
                cfg["buffer_size"],
                cfg["batch_size"],
                cfg["classes"],
            ],
            dtype=np.uint8,
            buffer=self.buff_labels.buf
        )

        for index in range(cfg["buffer_size"]):
            self.writer_index_queue.put(index)
    
    def writeBuffer(self, images, labels):
        index = self.writer_index_queue.get()
        np.copyto(self.buff_images_np[index], images)
        np.copyto(self.buff_labels_np[index], labels)
        self.reader_index_queue.put(index)
    
    def readBuffer(self):
        index = self.reader_index_queue.get()
        image = np.copy(self.buff_images_np[index])
        label = np.copy(self.buff_labels_np[index])
        self.writer_index_queue.put(index)

        return image, label

    def close(self):
        self.buff_images.close()
        self.buff_labels.close()
        self.buff_images.unlink()
        self.buff_labels.unlink()

class DataLoader():
    def __init__(self, images: str, cfg: dict, val=True):
        super().__init__()
        random.seed(time.time())

        self.cfg = copy.deepcopy(cfg)
        self.images, classes_name = load_filelist(images, self.cfg["file_checkers"], self.cfg["file_checker_bypass"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        if cfg["data_length"] in (tuple, list):
            self.data_length = cfg["data_length"][::-1][int(val)]
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

        if not val:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    True,
                    classes_name,
                ) for _ in range(loaders)
            ]

        else:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    False,
                    classes_name
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
        images = tf.convert_to_tensor(images, tf.float32) / 255.
        labels = tf.convert_to_tensor(labels, tf.float32)

        return images, labels