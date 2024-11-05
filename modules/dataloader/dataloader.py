import time
import random
import numpy as np
import math
import cv2
import multiprocessing
import gc
import queue

import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .utils import load_filelist, resize_contain, resize_stretch
from .augment import DataAugment

class Loader(multiprocessing.Process):
    def __init__(self, images: list, classes: int, batch_size: int, cfg: dict,):
        super().__init__()
        self.images = images
        self.queue = multiprocessing.Queue(cfg["queue_size"])
        self.cfg = cfg
        self.classes = classes
        self.data_length = len(images)
        self.batch_size = batch_size
        self.color_space = {"bgr": None, "rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}[cfg["color_space"].lower()]
        self.stop = False
    
    def _resize(self, image):
        if self.cfg["resize_method"] in ("default", "contain"):
            return resize_contain(image, self.cfg["image_size"])
        if self.cfg["resize_method"] in ("stretch",):
            return resize_stretch(image, self.cfg["image_size"])
        raise Exception("invalid resize method %s" % self.cfg["resize_method"])
    
    def _flip(self, image):
        if np.random.rand() < self.cfg["flip_vertical"]:
            image = image[::-1]
        
        if np.random.rand() < self.cfg["flip_horizontal"]:
            image = image[:, ::-1]
        
        return image
    
    def run(self):
        index = 0

        while not self.stop:
            if index == 0:
                random.shuffle(self.images)
            
            images_list = self.images[index * self.batch_size:(index + 1) * self.batch_size]
            index = (index + 1) % self.data_length

            if len(images_list) < self.batch_size:
                need = self.batch_size - len(images_list)
                images_list.extend(self.images[:need])

            images = np.zeros(
                [
                    self.batch_size,
                    self.cfg["image_size"],
                    self.cfg["image_size"],
                    3,
                ], dtype=np.float32
            )
            
            labels = np.zeros(
                [
                    self.batch_size,
                    self.classes
                ], dtype=np.uint8
            )

            for i, (image, label) in enumerate(images_list):
                image = cv2.imread(image, cv2.IMREAD_COLOR)
                image = self._flip(image)

                image = self._resize(image)

                if self.color_space is not None:
                    image = cv2.cvtColor(image, self.color_space)
                
                images[i] = image
                labels[i][label] = 1
            
            self.queue.put([images / 255., labels])

            del (
                images,
                labels,
            )
    
    def getData(self):
        return self.queue.get()

class DataLoader(Sequence):
    def __init__(self, images: list, cfg: dict, augment_flag=True):
        super().__init__()
        random.seed(time.time())

        if multiprocessing.parent_process() is not None:
            tf.config.set_visible_devices([], "GPU")
            cpus = tf.config.list_physical_devices("CPU")
            tf.config.set_visible_devices(cpus, "CPU")

        self.cfg = cfg
        self.augment_flag = augment_flag
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"]) if not self.augment_flag or self.cfg["data_length"] is None else self.cfg["data_length"]
        self.augment_rotate = 0
        self.data_index = 0

        self.subdivisions = self.cfg["subdivisions"] if self.cfg["subdivisions"] < self.cfg["batch_size"] else self.cfg["batch_size"]

        if augment_flag:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    self.cfg["batch_size"] // self.subdivisions,
                    cfg
                ) for _ in range(self.cfg["loaders"])
            ]

        else:
            self.subdivisions = 2
            length = len(images) // self.subdivisions
            random.shuffle(self.images)
            self.augments = [
                Loader(
                    self.images[i * length:(i + 1) * length],
                    self.classes,
                    self.cfg["batch_size"] // self.subdivisions,
                    cfg
                ) for i in range(self.subdivisions)
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
        images = []
        labels = []

        for _ in range(self.subdivisions):
            image, label = self.augments[self.augment_rotate].getData()
            self.augment_rotate = (self.augment_rotate + 1) % len(self.augments)

            images.append(image)
            labels.append(label)
        
        images_np = np.vstack(images)
        labels_np = np.vstack(labels)

        images_tf = tf.constant(images_np)
        labels_tf = tf.constant(labels_np)

        del (
            images,
            labels,
            images_np,
            labels_np,
        )
        
        return images_tf, labels_tf