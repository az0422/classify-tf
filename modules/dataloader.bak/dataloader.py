import multiprocessing
import yaml
import time
import random
import numpy as np
import math
import queue
import os
import signal

from tensorflow.keras.utils import Sequence

from .utils import load_filelist
from .augment import DataAugment

class DataLoaderManager(multiprocessing.Process):
    def __init__(self, images, classes, cfg, queues):
        super().__init__()
        self.images = images
        self.classes = classes
        self.cfg = cfg
        self.queues = queues
        self.processes = multiprocessing.Queue()
    
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
    
    def run(self):
        processes = queue.Queue()
        for num in range(self.cfg["loaders"]):
            augment = self._make_process(self.queues[num])
            processes.put(augment)
            self.processes.put(augment.pid)
        
        num = -1
        while True:
            num = (num + 1) % self.cfg["loaders"]

            process = processes.get()
            process.join()
            process.close()
            _ = self.processes.get()

            augment = self._make_process(self.queues[num])
            processes.put(augment)
            self.processes.put(augment.pid)
    
    def terminate(self):
        super().terminate()

        # kill confirm
        while not self.processes.empty():
            pid = self.processes.get()
            try:
                os.kill(pid, signal.SIGTERM)
            except:
                pass
        
class DataLoader(Sequence):
    def __init__(self, images, cfg):
        super().__init__()
        random.seed(time.time())
        self.cfg = cfg
        images, labels, classes_name = load_filelist(images, self.cfg["file_checkers"])
        self.classes_name = classes_name

        self.images = list(zip(images, labels))
        self.data_length = math.ceil(len(images) / self.cfg["batch_size"]) if self.cfg["data_length"] is None else self.cfg["data_length"]
        self.queues = [multiprocessing.Manager().Queue(self.cfg["queue_size"]) for _ in range(self.cfg["loaders"])]
        self.manager = DataLoaderManager(self.images, len(classes_name), self.cfg, self.queues)
        self.queue_index = 0
    
    def startAugment(self):
        self.manager.start()
    
    def stopAugment(self):
        for queue in self.queues:
            queue.close()
        self.manager.terminate()
        self.manager.join()
        self.manager.close()

    def __len__(self):
        return self.data_length
    
    def __getitem__(self, index=0):
        image, label = self.queues[self.queue_index].get()
        self.queue_index = (self.queue_index + 1) % self.cfg["loaders"]

        return image, label