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
    def __init__(self, images: list, classes: int, cfg: dict,):
        super().__init__()
        self.images = images
        self.queue = multiprocessing.Queue(cfg["queue_size"])
        self.cfg = cfg
        self.classes = classes
        self.data_length = len(images)
        self.color_space = {"bgr": None, "rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}[cfg["color_space"].lower()]
        self.stop = False
    
    def _resize(self, image):
        if self.cfg["resize_method"] in ("default", "contain"):
            return resize_contain(image, self.cfg["image_size"])
        if self.cfg["resize_method"] in ("stretch",):
            return resize_stretch(image, self.cfg["image_size"])
        raise Exception("invalid resize method %s" % self.cfg["resize_method"])
    
    def run(self):
        while not self.stop:
            images = []
            labels = []
            images_list = [random.choice(self.images) for _ in range(self.cfg["batch_size"])]

            for image, label in images_list:
                if image.endswith(".npy"):
                    image = np.load(image)
                else:
                    image = cv2.imread(image, cv2.IMREAD_COLOR)

                image = self._resize(image)

                if self.color_space is not None:
                    image = cv2.cvtColor(image, self.color_space)
                
                label_np = np.zeros([self.classes], dtype=np.uint8)
                label_np[label] = 1

                images.append(image[None, ...])
                labels.append(label_np[None, ...])
            
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            self.queue.put([images, labels])
    
    def getData(self):
        return self.queue.get()

class DataLoader(Sequence):
    def __init__(self, images: str, cfg: dict, augment_flag=True):
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

        if augment_flag:
            self.augments = [
                DataAugment(
                    random.randrange(-2**31, 2**31 - 1),
                    self.images,
                    self.classes,
                    cfg
                ) for _ in range(self.cfg["loaders"])
            ]

        else:
            self.augments = [
                Loader(
                    self.images,
                    self.classes,
                    cfg
                ) for i in range(self.cfg["loaders"])
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
        
        return images_tf, labels_tf