import time
import random
import math
import numpy as np
import multiprocessing

import tensorflow as tf

from multiprocessing import shared_memory

from .utils import load_filelist
from .augment import DataAugment

class DataLoader():
    def __init__(self, images: str, cfg: dict, augment_flag=True):
        super().__init__()
        random.seed(time.time())

        self.cfg = cfg
        self.augment_flag = augment_flag
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"] * cfg["buffer2_size"]) if not self.augment_flag or self.cfg["data_length"] is None else self.cfg["data_length"]

        self.augment_rotate = 0

        if augment_flag:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    True,
                    classes_name,
                ) for _ in range(self.cfg["loaders"])
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
                ) for _ in range(self.cfg["loaders"])
            ]
    
        self.reader_index_queue = multiprocessing.Queue(self.cfg["buffer_size"])
        self.writer_index_queue = multiprocessing.Queue(self.cfg["buffer_size"])

        self.buff_images = shared_memory.SharedMemory(
            create=True,
            size=self.cfg["buffer_size"] * self.cfg["batch_size"] * (self.cfg["image_size"] ** 2) * 3
        )

        self.buff_labels = shared_memory.SharedMemory(
            create=True,
            size=self.cfg["buffer_size"] * self.cfg["batch_size"] * self.classes
        )

        self.buff_images_np = np.ndarray(
            [
                self.cfg["buffer_size"],
                self.cfg["batch_size"],
                self.cfg["image_size"],
                self.cfg["image_size"],
                3,
            ],
            dtype=np.uint8,
            buffer=self.buff_images.buf
        )

        self.buff_labels_np = np.ndarray(
            [
                self.cfg["buffer_size"],
                self.cfg["batch_size"],
                self.classes,
            ],
            dtype=np.uint8,
            buffer=self.buff_labels.buf
        )

        for index in range(self.cfg["buffer_size"] - 1):
            self.writer_index_queue.put(index)
        
        for augment in self.augments:
            augment.set_buffer(self.buff_images, self.buff_labels, self.reader_index_queue, self.writer_index_queue)

    def startAugment(self):
        for augment in self.augments:
            augment.start()
    
    def stopAugment(self):
        for augment in self.augments:
            augment.terminate()
            augment.join()
        
        self.buff_images.close()
        self.buff_labels.close()
        self.buff_images.unlink()
        self.buff_labels.unlink()

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        index = self.reader_index_queue.get()
        image = tf.convert_to_tensor(self.buff_images_np[index], tf.float32) / 255.
        label = tf.convert_to_tensor(self.buff_labels_np[index], tf.float32)
        self.writer_index_queue.put(index)

        return image, label