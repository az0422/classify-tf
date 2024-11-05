import os
from typing import Union
import queue

import cv2
import numpy as np
import threading

def resize_stretch(image: np.ndarray, target_size=640):
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

def resize_contain(image: np.ndarray, target_size=640):
    height, width, _ = image.shape

    scale = target_size / max(height, width)
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

    pad_width = target_size - scaled_width
    pad_height = target_size - scaled_height

    top, bottom = pad_height // 2, pad_height - pad_height // 2
    left, right = pad_width // 2, pad_width - pad_width // 2

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image

def checker_log(ok: int, none: int):
    print("checking files... ok: %-16d\terror: %-16d" % (ok, none), end="\r")

def load_filelist(image: str, loaders: int):
    images = []
    labels = []

    images_test = []
    test_queue = queue.Queue()

    categories = sorted(os.listdir(image))

    for id, category in enumerate(categories):
        category_path = os.path.join(image, category)
        if os.path.isfile(category_path): continue

        files = os.listdir(category_path)
        for file in files:
            file_path = os.path.join(image, category, file)
            if os.path.isdir(file_path): continue
            images_test.append([file_path, id])
    
    none = 0
    ok = 0
    
    for mass in range(0, len(images_test), loaders):
        threads: list[threading.Thread] = []
        for image, label in images_test[mass:mass + loaders]:
            threads.append(TestImage(image, label, test_queue))
            threads[-1].start()
        
        for thread in threads:
            thread.join()
        
        while not test_queue.empty():
            image, label, flag = test_queue.get()
            if flag:
                ok += 1
                images.append(image)
                labels.append(label)
            else:
                none += 1
            
            checker_log(ok, none)
    
    print()
    return images, labels, categories

class TestImage(threading.Thread):
    def __init__(self, image: Union[str, bytes, bytearray, np.ndarray], label: int, queue: queue.Queue):
        super().__init__()
        self.image = image
        self.label = label
        self.queue = queue
    
    def run(self):
        if type(self.image) in (bytes, bytearray):
            image_np = np.frombuffer(self.image, np.uint8)
            test = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        elif type(self.image) is str:
            test = cv2.imread(self.image, cv2.IMREAD_COLOR)
        elif type(self.image) is np.ndarray:
            test = cv2.imread(self.image, cv2.IMREAD_COLOR)
        
        self.queue.put([self.image, self.label, test is not None])

class LoadImage(threading.Thread):
    def __init__(self, image: Union[str, bytes, bytearray, np.ndarray], label: int, queue: queue.Queue, image_size: int, resize_method: str):
        super().__init__()
        self.image = image
        self.label = label
        self.queue = queue
        self.image_size = image_size
        self.resize_method = resize_method
    
    def _resize(self, image):
        if self.resize_method in ("default", "contain"):
            return resize_contain(image, self.image_size)
        if self.resize_method in ("stretch",):
            return resize_stretch(image, self.image_size)
        raise Exception("invalid resize method %s" % self.resize_method)
    
    def run(self):
        if type(self.image) in (bytes, bytearray):
            image_np = np.frombuffer(self.image, np.uint8)
            image = self._resize(cv2.imdecode(image_np, cv2.IMREAD_COLOR))
        elif type(self.image) is str:
            image = self._resize(cv2.imread(self.image, cv2.IMREAD_COLOR))
        elif type(self.image) is np.ndarray:
            image = self._resize(self.image)
        
        self.queue.put([image, self.label])