import multiprocessing
import time
import random
import numpy as np
import math
import queue
import cv2
import os
import signal

from tensorflow.keras.utils import Sequence

from .utils import load_filelist, LoadImage
from .augment import DataAugment

class Loader(multiprocessing.Process):
    def __init__(self, images, queue, cfg, classes, data_length):
        super().__init__()
        self.images = images
        self.queue = queue
        self.cfg = cfg
        self.classes = classes
        self.data_length = data_length
        self.color_space = {"bgr": None, "rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV}[cfg["color_space"].lower()]
    
    def run(self):
        index = 0
        batch_size = self.cfg["batch_size"]
        while True:
            if index == 0:
                random.shuffle(self.images)
            
            images_list = self.images[index * batch_size:(index + 1) * batch_size]
            index = (index + 1) % self.data_length

            if len(images_list) < batch_size:
                need = batch_size - len(images_list)
                images_list.extend(self.images[:need])
            
            taked_images = [None for _ in range(batch_size)]
            taked_labels = [None for _ in range(batch_size)]
            threads = []
            
            for i, (image, label) in enumerate(images_list):
                threads.append(
                    LoadImage(image, taked_images, self.cfg["image_size"], i, self.cfg["resize_method"], np.float32)
                )
                threads[-1].start()
                taked_labels[i] = label
            
            for thread in threads:
                thread.join()
            
            images = np.zeros(
                [
                    batch_size,
                    self.cfg["image_size"],
                    self.cfg["image_size"],
                    3,
                ], dtype=self.cfg["data_type"]
            )
            labels = np.zeros(
                [
                    batch_size,
                    self.classes,
                ], dtype=np.uint8,
            )

            for i, (image, label) in enumerate(zip(taked_images, taked_labels)):
                images[i] = image.astype(self.cfg["data_type"]) / 255.
                labels[i][label] = 1
            
            self.queue.put([images, labels])

class DataLoader(Sequence):
    def __init__(self, images, cfg, augment_flag=False):
        super().__init__()
        random.seed(time.time())
        self.cfg = cfg
        self.augment_flag = augment_flag
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name
        self.classes = len(classes_name)

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"]) if not self.augment_flag or self.cfg["data_length"] is None else self.cfg["data_length"]
        if self.augment_flag:
            self.queues = [multiprocessing.Queue(self.cfg["queue_size"]) for _ in range(self.cfg["loaders"])]
            self.loaders = self.cfg["loaders"]
        else:
            self.queues = [multiprocessing.Queue(self.cfg["queue_size"])]
            self.loaders = 1
        self.augments = []
        self.queue_index = 0
    
    def _make_process(self, queue):
        augment = DataAugment(
            seed=random.randrange(-2147483648, 2147483647),
            flip_vertical=self.cfg["flip_vertical"],
            flip_horizontal=self.cfg["flip_horizontal"],
            rotate_degree=self.cfg["rotate_degree"],
            zoom=self.cfg["zoom"],
            translate_vertical=self.cfg["translate_vertical"],
            translate_horizontal=self.cfg["translate_horizontal"],
            hsv_h=self.cfg["hsv_h"],
            hsv_s=self.cfg["hsv_s"],
            hsv_v=self.cfg["hsv_v"],
            noise=self.cfg["noise"],
            dequality=self.cfg["dequality"],
            crop_ratio=self.cfg["crop_ratio"],
        )
        augment.data(
            self.images,
            self.classes,
            queue,
            batch_size=self.cfg["batch_size"],
            image_size=self.cfg["image_size"],
            resize_method=self.cfg["resize_method"],
            color_space=self.cfg["color_space"],
            dtype=self.cfg["data_type"],
        )
        augment.start()
        return augment
    
    def startAugment(self):
        if self.augment_flag:
            self.augments = [self._make_process(q) for q in self.queues]
        else:
            self.augments = [Loader(self.images, self.queues[0], self.cfg, self.classes, self.data_length)]
            self.augments[0].start()
    
    def stopAugment(self):
        for augment in self.augments:
            augment.terminate()
            augment.join()
            augment.close()

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        image, label = self.queues[self.queue_index].get()
        self.queue_index = (self.queue_index + 1) % self.loaders

        return image, label