import time
import random
import math

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .utils import load_filelist
from .augment import DataAugment

class DataLoader(Sequence):
    def __init__(self, images: str, cfg: dict, aux_multiply: int, augment_flag=True):
        super().__init__()
        random.seed(time.time())

        self.cfg = cfg
        self.augment_flag = augment_flag
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name
        self.classes = len(classes_name)
        self.aux_multiply = aux_multiply

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"]) if not self.augment_flag or self.cfg["data_length"] is None else self.cfg["data_length"]
        self.augment_rotate = 0
        self.data_index = 0

        if augment_flag:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    True,
                ) for _ in range(self.cfg["loaders"])
            ]

        else:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg,
                    False
                ) for _ in range(self.cfg["loaders"])
            ]
    
    def startAugment(self):
        for augment in self.augments:
            augment.start()
    
    def stopAugment(self):
        for augment in self.augments:
            augment.terminate()
    
    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        images, labels = self.augments[self.augment_rotate].getData()
        self.augment_rotate = (self.augment_rotate + 1) % len(self.augments)

        images_tf = tf.convert_to_tensor(images, tf.float32) / 255.
        labels_tf = tf.convert_to_tensor(labels, tf.float32)

        if self.aux_multiply > 1:
            labels_tf = tf.expand_dims(labels_tf, axis=1)
            labels_tf = tf.tile(labels_tf, [1, self.aux_multiply, 1])
        
        return images_tf, labels_tf