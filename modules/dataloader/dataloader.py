import multiprocessing
import yaml
import time
import math
import numpy as np
import os

from tensorflow.keras.utils import Sequence

from .utils import load_filelist, LoadImage
from .augment import DataAugment

class DataLoaderManager(multiprocessing.Process):
    def __init__(self, images, labels, cfg, results, lock):
        super().__init__()
        self.images = []
        self.cfg = cfg
        self.stop_flag = False
        self.results = results
        self.lock = lock

        for image, label in zip(images, labels):
            self.images.append([image, label])
    
    def stop_sig(self):
        self.stop_flag = True
    
    def run(self):
        threads = []
        for _ in range(self.cfg["loaders"] - 1):
            augment = DataAugment(
                seed=int(np.random.rand() * 65535),
                flip_vertical=self.cfg["flip_vertical"],
                flip_horizontal=self.cfg["flip_horizontal"],
                rotate_degree=self.cfg["rotate_degree"],
                zoom=self.cfg["zoom"],
                translate_vertical=self.cfg["translate_vertical"],
                translate_horizontal=self.cfg["translate_horizontal"],
                hsv_h=self.cfg["hsv_h"],
                hsv_s=self.cfg["hsv_s"],
                hsv_v=self.cfg["hsv_v"],
                noise=self.cfg["noise"]
            )
            augment.data(self.images, self.results, self.lock, self.cfg["batch_size"], self.cfg["image_size"])
            augment.start()
            threads.insert(0, augment)
        
        while True:
            if self.stop_flag: break

            if self.results.qsize() > self.cfg["data_length"] * 2:
                time.sleep(5e-3)
                continue

            thread = threads.pop()
            thread.join()
            thread.close()
            
            augment = DataAugment(
                seed=int(np.random.rand() * 65535),
                flip_vertical=self.cfg["flip_vertical"],
                flip_horizontal=self.cfg["flip_horizontal"],
                rotate_degree=self.cfg["rotate_degree"],
                zoom=self.cfg["zoom"],
                translate_vertical=self.cfg["translate_vertical"],
                translate_horizontal=self.cfg["translate_horizontal"],
                hsv_h=self.cfg["hsv_h"],
                hsv_s=self.cfg["hsv_s"],
                hsv_v=self.cfg["hsv_v"],
                noise=self.cfg["noise"]
            )
            augment.data(self.images, self.results, self.lock, self.cfg["batch_size"], self.cfg["image_size"])
            augment.start()
            threads.insert(0, augment)
        
        for thread in threads:
            thread.join()
            thread.close()

class DataLoader(Sequence):
    def __init__(self, images, image_size, batch_size, cfg):
        self.images, self.labels, self.categories = load_filelist(images)

        if not os.path.isdir(cfg):
            cfg = os.path.join("cfg", cfg)
        
        self.cfg = yaml.full_load(open(cfg, "r"))
        self.cfg["image_size"] = image_size
        self.cfg["batch_size"] = batch_size
        self.cfg["data_length"] = math.ceil(len(self.images) / self.cfg["batch_size"])

        self.results = multiprocessing.Manager().Queue()
        self.lock = multiprocessing.Lock()
        self.manager = DataLoaderManager(self.images, self.labels, self.cfg, self.results, self.lock)
    
    def getCategories(self):
        return self.categories
    
    def startAugment(self):
        self.manager.start()
    
    def stopAugment(self):
        self.manager.terminate()
        self.manager.join()
        self.manager.close()
    
    def __len__(self):
        return self.cfg["data_length"]
    
    def __getitem__(self, index=0):
        while True:
            if self.results.qsize() > 0: break
            time.sleep(1e-3)

        self.lock.acquire()
        image, label = self.results.get()
        self.lock.release()
        
        return image, label

class DataLoaderVal(Sequence):
    def __init__(self, images, image_size, batch_size):
        self.images, self.labels, self.categories = load_filelist(images)
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.images) / self.batch_size)
    
    def __getitem__(self, index=0):
        images = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        images_len = len(images)
        images.extend(images[:self.batch_size - images_len])
        labels.extend(labels[:self.batch_size - images_len])

        images_load = [None for _ in range(self.batch_size)]
        threads = []
        for index, (image) in enumerate(images):
            threads.append(LoadImage(image, images_load, self.image_size, index))
            threads[-1].start()
        
        for thread in threads:
            thread.join()
        
        images_np = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.float32)
        labels_np = np.zeros([self.batch_size], dtype=np.float32)

        for index, (image, label) in enumerate(zip(images_load, labels)):
            images_np[index] = image
            labels_np[index] = label
        
        return images_np, labels_np